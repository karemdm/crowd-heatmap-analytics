"""Tests for OccupancyStats."""

from crowd_heatmap.stats import OccupancyStats


def test_initial_state_is_zero():
    stats = OccupancyStats()
    assert stats.current == 0
    assert stats.peak == 0
    assert stats.unique_count == 0


def test_current_peak_and_unique():
    stats = OccupancyStats()
    stats.update([1, 2, 3])
    assert (stats.current, stats.peak, stats.unique_count) == (3, 3, 3)
    stats.update([2, 4])
    assert (stats.current, stats.peak, stats.unique_count) == (2, 3, 4)
    stats.update([])
    assert (stats.current, stats.peak, stats.unique_count) == (0, 3, 4)


def test_peak_grows_only():
    stats = OccupancyStats()
    for ids in ([1], [1, 2, 3, 4, 5], [1, 2]):
        stats.update(ids)
    assert stats.peak == 5


def test_accepts_generators():
    stats = OccupancyStats()
    stats.update(i for i in range(4))
    assert stats.current == 4
    assert stats.unique_count == 4
