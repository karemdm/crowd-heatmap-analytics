"""Tests for the drawing helpers."""

import numpy as np

from crowd_heatmap.rendering import (
    draw_hud, draw_titles, draw_track, resize_for_display, side_by_side)
from crowd_heatmap.stats import OccupancyStats


def test_draw_track_marks_the_box_and_handles_short_trails():
    img = np.zeros((100, 100, 3), np.uint8)
    draw_track(img, (10, 20, 60, 80), 1, 0.9, [])
    assert tuple(img[50, 10]) == (0, 255, 0)  # left edge of the box, BGR green
    draw_track(img, (10, 20, 60, 80), 1, 0.9, [(5, 5)])


def test_draw_track_box_at_frame_corner_does_not_crash():
    img = np.zeros((50, 50, 3), np.uint8)
    draw_track(img, (0, 0, 50, 50), 2, 0.5, [(0, 0), (49, 49)])


def test_hud_darkens_only_its_box_and_survives_small_images():
    img = np.full((200, 400, 3), 200, np.uint8)
    draw_hud(img, OccupancyStats())
    assert img[150, 350, 0] == 200
    assert img[110, 310, 0] < 200
    draw_hud(np.full((30, 40, 3), 200, np.uint8), OccupancyStats())


def test_titles_on_narrow_images():
    draw_titles(np.zeros((60, 100, 3), np.uint8), np.zeros((60, 100, 3), np.uint8))


def test_side_by_side_doubles_width():
    a = np.zeros((10, 20, 3), np.uint8)
    assert side_by_side(a, a).shape == (10, 40, 3)


def test_resize_for_display():
    img = np.zeros((100, 200, 3), np.uint8)
    assert resize_for_display(img, 1.0) is img
    assert resize_for_display(img, 2.0) is img
    assert resize_for_display(img, 0.5).shape == (50, 100, 3)
    assert resize_for_display(img, 0.0).shape == (1, 1, 3)
