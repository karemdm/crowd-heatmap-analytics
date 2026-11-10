"""Configuration dataclass holding every tunable parameter and its default."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Runtime configuration for the tracking + heatmap pipeline.

    Defaults reproduce the behaviour of the original prototype.

    Attributes:
        model: Ultralytics weights path or official model name (auto-downloaded).
        conf: Detection confidence threshold (low to catch small, distant people).
        imgsz: Inference image size; high values preserve small people.
        tracker: Ultralytics tracker configuration file.
        classes: COCO class IDs to track (0 = person).
        decay: Per-frame multiplicative decay of the accumulated heat.
        intensity: Value painted inside each foot-point circle on the frame mask.
        radius: Radius in pixels of the foot-point circle.
        blur: Gaussian blur kernel size (odd, square) applied to the frame mask.
        alpha: Weight of the original frame in the heatmap blend; the colormap
            gets ``1 - alpha``.
        trail: Number of most recent foot points kept per track for the trail.
        trail_max_age: Frames a track may go unseen before its trail is
            dropped from memory. Well above ByteTrack's default lost-track
            buffer (30 frames), so visible trails are not affected.
        display: Whether to show a live preview window.
        screen_fraction_w: Maximum preview width as a fraction of screen width.
        screen_fraction_h: Maximum preview height as a fraction of screen height.
        window_name: Title of the preview window.
        fourcc: Codec used by the output video writer.
        save_heatmap: Optional path to save the final accumulated heatmap image.
    """

    model: str = "yolo26x.pt"
    conf: float = 0.10
    imgsz: int = 1920
    tracker: str = "bytetrack.yaml"
    classes: List[int] = field(default_factory=lambda: [0])
    decay: float = 0.995
    intensity: float = 3.5
    radius: int = 18
    blur: int = 35
    alpha: float = 0.6
    trail: int = 30
    trail_max_age: int = 300
    display: bool = True
    screen_fraction_w: float = 0.90
    screen_fraction_h: float = 0.85
    window_name: str = "Tracking & Heatmap"
    fourcc: str = "mp4v"
    save_heatmap: Optional[str] = None
