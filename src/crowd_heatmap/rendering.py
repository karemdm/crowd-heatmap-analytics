"""Drawing helpers: tracks, HUD, panel titles and side-by-side composition."""

from typing import Sequence, Tuple

import cv2
import numpy as np

from crowd_heatmap.layout import scaled_size
from crowd_heatmap.stats import OccupancyStats

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def draw_track(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    track_id: int,
    conf: float,
    trail: Sequence[Tuple[int, int]],
) -> None:
    """Draws a trajectory polyline, bounding box and ID label in place.

    Args:
        image: BGR image to draw on.
        box: ``(x1, y1, x2, y2)`` in pixels.
        track_id: Tracker ID.
        conf: Detection confidence.
        trail: Foot points of the track, oldest first.
    """
    x1, y1, x2, y2 = box
    if trail:
        pts = np.asarray(trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=YELLOW, thickness=2)
    cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)
    cv2.putText(image, f"ID: {track_id} ({conf:.2f})", (x1, y1 - 8), FONT, 0.5, GREEN, 2)


def draw_hud(image: np.ndarray, stats: OccupancyStats) -> None:
    """Draws the semi-transparent occupancy HUD in the top-left corner in place.

    Args:
        image: BGR image to draw on.
        stats: Current occupancy statistics.
    """
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (320, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, f"Current Occupancy: {stats.current}", (20, 35), FONT, 0.6, WHITE, 2)
    cv2.putText(image, f"Peak Occupancy: {stats.peak}", (20, 65), FONT, 0.6, YELLOW, 2)
    cv2.putText(
        image, f"Total Unique People: {stats.unique_count}", (20, 95), FONT, 0.6, GREEN, 2
    )


def draw_titles(tracking: np.ndarray, heatmap: np.ndarray) -> None:
    """Writes the panel titles on both images in place.

    Args:
        tracking: Left (tracking) panel.
        heatmap: Right (heatmap) panel.
    """
    cv2.putText(tracking, "ORIGINAL TRACKING", (20, 40), FONT, 0.9, GREEN, 2)
    x = max(20, heatmap.shape[1] - 260)
    cv2.putText(heatmap, "SPATIAL HEATMAP", (x, 40), FONT, 0.9, RED, 2)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Concatenates two same-height images horizontally."""
    return np.hstack((left, right))


def resize_for_display(image: np.ndarray, ratio: float) -> np.ndarray:
    """Downscales an image by ``ratio`` with area interpolation.

    Args:
        image: BGR image.
        ratio: Scale ratio from :func:`crowd_heatmap.layout.fit_scale`.

    Returns:
        The resized image, or the input itself when ``ratio`` is 1.0 or more.
    """
    if ratio >= 1.0:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, scaled_size(w, h, ratio), interpolation=cv2.INTER_AREA)
