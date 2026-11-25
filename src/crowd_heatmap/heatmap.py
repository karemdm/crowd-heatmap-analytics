"""Decaying spatial heatmap accumulator (numpy + OpenCV)."""

import math
from typing import Iterable, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]

REFERENCE_FPS = 30.0
"""Frame rate at which ``decay`` is defined as a per-frame factor."""


def effective_decay(decay: float, fps: float = REFERENCE_FPS) -> float:
    """Converts the reference per-frame decay into the per-frame factor at ``fps``.

    ``decay`` is the per-frame factor at :data:`REFERENCE_FPS` (30 FPS), so the
    half-life in seconds is the same for every frame rate: ``decay ** (30 / fps)``.

    Args:
        decay: Per-frame decay at the reference frame rate, in ``[0, 1]``.
        fps: Real processing rate in frames per second (> 0).

    Returns:
        The multiplicative decay to apply on each processed frame.

    Raises:
        ValueError: If ``fps`` is not a positive finite number.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    return decay ** (REFERENCE_FPS / fps)


class HeatmapAccumulator:
    """Accumulates blurred foot-point heat over time with exponential decay.

    For each frame a float mask is filled with circles of value ``intensity``
    at the foot points, blurred with a square Gaussian kernel, and added to the
    decayed accumulator: ``H_t = frame_decay * H_{t-1} + blur(M_t)``, where
    ``frame_decay = decay ** (30 / fps)``.

    The mask buffer is allocated once and reused. Only the region that can be
    non-zero after blurring (the bounding box of the circles, grown by half the
    kernel size plus one pixel) is drawn, blurred and added. This gives the same
    result as blurring the full frame (up to float rounding), because the blur
    of an all-zero area is zero; frames without points only decay.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        decay: Per-frame decay factor at 30 FPS (reference). At another frame
            rate the per-frame factor is ``decay ** (30 / fps)``, so the half-life
            in seconds does not depend on the FPS.
        intensity: Value painted inside each circle.
        radius: Circle radius in pixels.
        blur: Gaussian kernel size (odd, positive).
        fps: Real processing rate in frames per second (> 0).

    Raises:
        ValueError: If the frame size is not positive, ``decay`` is outside
            ``[0, 1]``, ``radius`` is negative, ``intensity`` is not finite, or
            ``blur`` is not a positive odd integer, or ``fps`` is not a
            positive finite number.
    """

    def __init__(
        self,
        width: int,
        height: int,
        decay: float = 0.995,
        intensity: float = 3.5,
        radius: int = 18,
        blur: int = 35,
        fps: float = REFERENCE_FPS,
    ) -> None:
        """Allocates a zeroed accumulator and a reusable mask buffer."""
        if width < 1 or height < 1:
            raise ValueError("frame size must be positive")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if radius < 0:
            raise ValueError("radius must be >= 0")
        if not math.isfinite(intensity):
            raise ValueError("intensity must be finite")
        if blur < 1 or blur % 2 == 0:
            raise ValueError("blur kernel size must be a positive odd integer")
        self.decay = decay
        self.frame_decay = effective_decay(decay, fps)
        self.intensity = intensity
        self.radius = radius
        self.blur = blur
        self.accumulated = np.zeros((height, width), dtype=np.float32)
        self._mask = np.zeros((height, width), dtype=np.float32)

    def update(self, points: Iterable[Point]) -> None:
        """Adds one frame of foot points to the accumulator.

        Args:
            points: ``(x, y)`` foot points detected in the frame. May be empty;
                the accumulator still decays.
        """
        self.accumulated *= self.frame_decay
        pts = [(int(x), int(y)) for x, y in points]
        if not pts:
            return
        height, width = self.accumulated.shape
        # One extra pixel so the ROI-border reflection used by GaussianBlur never
        # reaches a circle pixel; the result then matches a full-frame blur.
        margin = self.radius + self.blur // 2 + 1
        x0 = max(0, min(x for x, _ in pts) - margin)
        y0 = max(0, min(y for _, y in pts) - margin)
        x1 = min(width, max(x for x, _ in pts) + margin + 1)
        y1 = min(height, max(y for _, y in pts) + margin + 1)
        if x0 >= x1 or y0 >= y1:
            return  # Every circle lies outside the frame.
        mask = self._mask[y0:y1, x0:x1]
        for x, y in pts:
            cv2.circle(mask, (x - x0, y - y0), self.radius, float(self.intensity), -1)
        self.accumulated[y0:y1, x0:x1] += cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        mask.fill(0.0)

    def normalized(self) -> np.ndarray:
        """Returns the accumulator min-max normalized to ``uint8`` 0-255."""
        norm = cv2.normalize(self.accumulated, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

    def colorized(self) -> np.ndarray:
        """Returns the normalized accumulator with the JET colormap (BGR)."""
        return cv2.applyColorMap(self.normalized(), cv2.COLORMAP_JET)

    def blend(self, frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Blends the colorized heatmap over a frame.

        Args:
            frame: BGR frame with the same size as the accumulator.
            alpha: Weight of ``frame``; the colormap gets ``1 - alpha``.

        Returns:
            The blended BGR image.
        """
        return cv2.addWeighted(frame, alpha, self.colorized(), 1 - alpha, 0)
