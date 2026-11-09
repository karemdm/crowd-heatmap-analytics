"""Occupancy statistics (standard library only)."""

from typing import Hashable, Iterable, Set


class OccupancyStats:
    """Tracks current occupancy, peak occupancy and unique track IDs.

    Note:
        ``unique_count`` counts distinct tracker IDs, not distinct people:
        a tracker ID switch (re-identification failure) inflates it.
    """

    def __init__(self) -> None:
        """Initializes all counters to zero."""
        self.current: int = 0
        self.peak: int = 0
        self._ids: Set[Hashable] = set()

    def update(self, track_ids: Iterable[Hashable]) -> None:
        """Registers the track IDs visible in the current frame.

        Args:
            track_ids: IDs of the tracks detected in this frame. An empty
                iterable sets current occupancy to zero.
        """
        ids = list(track_ids)
        self.current = len(ids)
        self.peak = max(self.peak, self.current)
        self._ids.update(ids)

    @property
    def unique_count(self) -> int:
        """Number of distinct track IDs seen so far."""
        return len(self._ids)
