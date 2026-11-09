"""Bounded per-track trajectory history (standard library only)."""

from collections import deque
from typing import Deque, Dict, Hashable, List, Tuple

Point = Tuple[int, int]


class TrajectoryStore:
    """Keeps the most recent ``max_length`` points of every track.

    Args:
        max_length: Maximum number of points retained per track (>= 1).

    Raises:
        ValueError: If ``max_length`` is smaller than 1.
    """

    def __init__(self, max_length: int = 30) -> None:
        """Creates an empty store."""
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self.max_length = max_length
        self._tracks: Dict[Hashable, Deque[Point]] = {}

    def add(self, track_id: Hashable, point: Point) -> List[Point]:
        """Appends a point to a track and returns the track's current trail.

        Args:
            track_id: Tracker ID.
            point: ``(x, y)`` pixel coordinates.

        Returns:
            The retained points of the track, oldest first.
        """
        trail = self._tracks.get(track_id)
        if trail is None:
            trail = deque(maxlen=self.max_length)
            self._tracks[track_id] = trail
        trail.append(point)
        return list(trail)

    def get(self, track_id: Hashable) -> List[Point]:
        """Returns the trail of a track (empty list if unknown)."""
        return list(self._tracks.get(track_id, ()))

    def __len__(self) -> int:
        """Number of tracks stored."""
        return len(self._tracks)
