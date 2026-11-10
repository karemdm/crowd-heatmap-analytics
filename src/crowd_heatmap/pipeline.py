"""Video processing loop: tracking, heatmap accumulation, rendering and output."""

import logging
import math
import os
from typing import List, Tuple

import cv2

from crowd_heatmap.config import Config
from crowd_heatmap.dialogs import get_screen_resolution
from crowd_heatmap.heatmap import HeatmapAccumulator
from crowd_heatmap.layout import fit_scale
from crowd_heatmap.rendering import (
    draw_hud,
    draw_titles,
    draw_track,
    resize_for_display,
    side_by_side,
)
from crowd_heatmap.stats import OccupancyStats
from crowd_heatmap.trajectories import TrajectoryStore

logger = logging.getLogger(__name__)

Detection = Tuple[Tuple[int, int, int, int], int, float]
FALLBACK_FPS = 30.0


def extract_detections(result) -> List[Detection]:
    """Converts an Ultralytics tracking result into plain Python tuples.

    Args:
        result: One ``ultralytics.engine.results.Results`` object.

    Returns:
        A list of ``((x1, y1, x2, y2), track_id, conf)``. Detections without a
        track ID (not yet confirmed by the tracker) are skipped.
    """
    boxes = result.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    return [
        (tuple(int(v) for v in box), int(tid), float(c))
        for box, tid, c in zip(xyxy, ids, confs)
    ]


def _display_ratio(cfg: Config, width: int, height: int) -> float:
    """Returns the preview scale ratio, or 1.0 when the screen size is unknown."""
    screen = get_screen_resolution()
    if screen is None:
        return 1.0
    return fit_scale(width, height, screen[0], screen[1],
                     cfg.screen_fraction_w, cfg.screen_fraction_h)


def process_video(input_path: str, output_path: str, cfg: Config) -> int:
    """Runs tracking and heatmap rendering on a video and writes the result.

    The output is a side-by-side video (tracking | heatmap) at twice the input
    width. Frames are written as they are processed, so stopping early with
    ``Q`` keeps everything up to that point.

    Args:
        input_path: Source video file.
        output_path: Destination video file.
        cfg: Pipeline configuration.

    Returns:
        Number of frames written.

    Raises:
        ValueError: If the input video cannot be opened.
        RuntimeError: If the output writer cannot be opened.
    """
    from ultralytics import YOLO  # Heavy import kept local to the pipeline.

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Could not read frame size of video source: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not math.isfinite(fps) or fps <= 0:
        logger.warning("Source FPS unavailable; using %.0f", FALLBACK_FPS)
        fps = FALLBACK_FPS

    # Load the model before creating the writer, so a bad --model leaves no empty file.
    try:
        model = YOLO(cfg.model)
    except Exception:
        cap.release()
        raise

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*cfg.fourcc), fps, (width * 2, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    ratio = _display_ratio(cfg, width * 2, height) if cfg.display else 1.0
    heat = HeatmapAccumulator(width, height, cfg.decay, cfg.intensity, cfg.radius, cfg.blur)
    trails = TrajectoryStore(cfg.trail, cfg.trail_max_age)
    stats = OccupancyStats()
    frames = 0

    logger.info("Model %s (conf=%.2f, imgsz=%d, tracker=%s)",
                cfg.model, cfg.conf, cfg.imgsz, cfg.tracker)
    logger.info("Writing output to %s", output_path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("End of video stream reached.")
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            result = model.track(frame, persist=True, tracker=cfg.tracker, conf=cfg.conf,
                                 classes=cfg.classes, imgsz=cfg.imgsz, verbose=False)[0]
            detections = extract_detections(result)
            stats.update(tid for _, tid, _ in detections)

            tracking = frame.copy()
            feet = []
            for (x1, y1, x2, y2), tid, conf in detections:
                foot = ((x1 + x2) // 2, y2)
                feet.append(foot)
                draw_track(tracking, (x1, y1, x2, y2), tid, conf, trails.add(tid, foot))
            trails.end_frame()
            heat.update(feet)

            heat_view = heat.blend(frame, cfg.alpha)
            draw_hud(heat_view, stats)
            draw_titles(tracking, heat_view)
            combined = side_by_side(tracking, heat_view)
            writer.write(combined)
            frames += 1

            if cfg.display:
                cv2.imshow(cfg.window_name, resize_for_display(combined, ratio))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                    logger.info("Stopped by user.")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        writer.release()
        if cfg.display:
            cv2.destroyAllWindows()
        logger.info("Saved %s (%d frames).", output_path, frames)

    if cfg.save_heatmap and frames > 0:
        heat_dir = os.path.dirname(os.path.abspath(cfg.save_heatmap))
        os.makedirs(heat_dir, exist_ok=True)
        if cv2.imwrite(cfg.save_heatmap, heat.colorized()):
            logger.info("Saved final heatmap to %s", cfg.save_heatmap)
        else:
            logger.error("Could not write heatmap image: %s", cfg.save_heatmap)
    return frames
