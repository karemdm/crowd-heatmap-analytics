"""Tests for TrajectoryStore."""

from crowd_heatmap.trajectories import TrajectoryStore


def test_add_returns_trail_in_order():
    store = TrajectoryStore(max_length=5)
    store.add(1, (0, 0))
    trail = store.add(1, (1, 1))
    assert trail == [(0, 0), (1, 1)]


def test_max_length_drops_oldest():
    store = TrajectoryStore(max_length=3)
    for i in range(6):
        trail = store.add(7, (i, i))
    assert trail == [(3, 3), (4, 4), (5, 5)]
    assert store.get(7) == trail


def test_tracks_are_independent():
    store = TrajectoryStore(max_length=30)
    store.add(1, (1, 1))
    store.add(2, (2, 2))
    assert store.get(1) == [(1, 1)]
    assert store.get(2) == [(2, 2)]
    assert store.get(99) == []
    assert len(store) == 2


def test_returned_trail_is_a_copy():
    store = TrajectoryStore(max_length=3)
    trail = store.add(1, (0, 0))
    trail.append((9, 9))
    assert store.get(1) == [(0, 0)]


def test_invalid_max_length():
    try:
        TrajectoryStore(max_length=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
