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
        """Allocates a zeroed accumulator."""
        if blur < 1 or blur % 2 == 0:
            raise ValueError("blur kernel size must be a positive odd integer")
        self.decay = decay
        self.intensity = intensity
        self.radius = radius
        self.blur = blur
        self.accumulated = np.zeros((height, width), dtype=np.float32)

    def update(self, points: Iterable[Point]) -> None:
        """Adds one frame of foot points to the accumulator.

        Args:
            points: ``(x, y)`` foot points detected in the frame. May be empty;
                the accumulator still decays.
        """
        mask = np.zeros_like(self.accumulated)
        for x, y in points:
            cv2.circle(mask, (int(x), int(y)), self.radius, float(self.intensity), -1)
        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        self.accumulated = self.accumulated * self.decay + mask

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
