"""Pure layout helpers for the preview window (standard library only)."""

from typing import Tuple


def fit_scale(
    frame_w: int,
    frame_h: int,
    screen_w: int,
    screen_h: int,
    frac_w: float = 0.90,
    frac_h: float = 0.85,
) -> float:
    """Computes the downscale ratio that fits a frame inside part of the screen.

    The target box is ``int(screen_w * frac_w)`` by ``int(screen_h * frac_h)``.
    The ratio never exceeds 1.0, so frames are never upscaled.

    Args:
        frame_w: Frame width in pixels (> 0).
        frame_h: Frame height in pixels (> 0).
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.
        frac_w: Fraction of the screen width available.
        frac_h: Fraction of the screen height available.

    Returns:
        Scale ratio in ``(0, 1]`` (or 0 for a degenerate screen size).

    Raises:
        ValueError: If a frame dimension is not positive.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame dimensions must be positive")
    max_w = int(screen_w * frac_w)
    max_h = int(screen_h * frac_h)
    return max(0.0, min(1.0, max_w / frame_w, max_h / frame_h))


def scaled_size(frame_w: int, frame_h: int, ratio: float) -> Tuple[int, int]:
    """Applies a scale ratio to a size, keeping at least 1 pixel per side.

    Args:
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        ratio: Scale ratio.

    Returns:
        ``(width, height)`` after scaling.
    """
    return max(1, int(frame_w * ratio)), max(1, int(frame_h * ratio))
