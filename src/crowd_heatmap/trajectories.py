"""Bounded per-track trajectory history (standard library only)."""

from collections import deque
from typing import Deque, Dict, Hashable, List, Optional, Tuple

Point = Tuple[int, int]


class TrajectoryStore:
    """Keeps the most recent ``max_length`` points of every track.

    Trails of tracks that have gone ``max_age`` consecutive frames without a
    point are dropped by :meth:`end_frame`, so memory does not grow with the
    number of tracks that have left the scene.

    Args:
        max_length: Maximum number of points retained per track (>= 1).
        max_age: Frames a track may go unseen before its trail is dropped
            (>= 1), or ``None`` to keep every trail until the end of the run.

    Raises:
        ValueError: If ``max_length`` is smaller than 1 or ``max_age`` is smaller than 1.
    """

    def __init__(self, max_length: int = 30, max_age: Optional[int] = None) -> None:
        """Creates an empty store."""
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if max_age is not None and max_age < 1:
            raise ValueError("max_age must be >= 1 or None")
        self.max_length = max_length
        self.max_age = max_age
        self._tracks: Dict[Hashable, Deque[Point]] = {}
        self._last_seen: Dict[Hashable, int] = {}
        self._frame = 0

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
        self._last_seen[track_id] = self._frame
        return list(trail)

    def end_frame(self) -> None:
        """Advances the frame counter and drops trails older than ``max_age``."""
        if self.max_age is not None:
            stale = [tid for tid, seen in self._last_seen.items()
                     if self._frame - seen >= self.max_age]
            for tid in stale:
                del self._tracks[tid]
                del self._last_seen[tid]
        self._frame += 1

    def get(self, track_id: Hashable) -> List[Point]:
        """Returns the trail of a track (empty list if unknown)."""
        return list(self._tracks.get(track_id, ()))

    def __len__(self) -> int:
        """Number of tracks stored."""
        return len(self._tracks)
