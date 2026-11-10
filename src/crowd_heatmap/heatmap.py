"""Decaying spatial heatmap accumulator (numpy + OpenCV)."""

from typing import Iterable, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


class HeatmapAccumulator:
    """Accumulates blurred foot-point heat over time with exponential decay.

    For each frame a float mask is filled with circles of value ``intensity``
    at the foot points, blurred with a square Gaussian kernel, and added to the
    decayed accumulator: ``H_t = decay * H_{t-1} + blur(M_t)``.

    The mask buffer is allocated once and reused. Only the region that can be
    non-zero after blurring (the bounding box of the circles, grown by half the
    kernel size plus one pixel) is drawn, blurred and added. This gives the same
    result as blurring the full frame (up to float rounding), because the blur
    of an all-zero area is zero; frames without points only decay.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        decay: Multiplicative decay applied each frame.
        intensity: Value painted inside each circle.
        radius: Circle radius in pixels.
        blur: Gaussian kernel size (odd, positive).

    Raises:
        ValueError: If ``blur`` is not a positive odd integer.
    """

    def __init__(
        self,
        width: int,
        height: int,
        decay: float = 0.995,
        intensity: float = 3.5,
        radius: int = 18,
        blur: int = 35,
    ) -> None:
        """Allocates a zeroed accumulator and a reusable mask buffer."""
        if blur < 1 or blur % 2 == 0:
            raise ValueError("blur kernel size must be a positive odd integer")
        self.decay = decay
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
        self.accumulated *= self.decay
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
