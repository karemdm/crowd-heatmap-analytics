"""Tests for HeatmapAccumulator (synthetic data, numpy + OpenCV only)."""

import cv2
import numpy as np
import pytest

from crowd_heatmap.heatmap import HeatmapAccumulator, effective_decay


def _reference(width, height, steps, decay, intensity, radius, blur):
    """Full-frame implementation of H_t = decay * H_{t-1} + blur(mask_t)."""
    acc = np.zeros((height, width), np.float32)
    for pts in steps:
        acc *= decay
        mask = np.zeros((height, width), np.float32)
        for x, y in pts:
            cv2.circle(mask, (int(x), int(y)), radius, float(intensity), -1)
        acc += cv2.GaussianBlur(mask, (blur, blur), 0)
    return acc


def test_roi_update_matches_full_frame_blur():
    steps = [[(50, 40), (60, 45)], [], [(5, 5)], [(95, 69), (0, 0)], [(-3, 10), (30, 80)]]
    heat = HeatmapAccumulator(100, 70, 0.9, 3.5, 6, 11)
    for pts in steps:
        heat.update(pts)
    ref = _reference(100, 70, steps, 0.9, 3.5, 6, 11)
    assert np.allclose(heat.accumulated, ref, atol=1e-5)


def test_no_points_only_decays():
    heat = HeatmapAccumulator(40, 30, decay=0.5, radius=3, blur=5)
    heat.update([(20, 15)])
    before = heat.accumulated.copy()
    heat.update([])
    assert np.allclose(heat.accumulated, before * 0.5)


def test_points_outside_the_frame_are_ignored():
    heat = HeatmapAccumulator(40, 30, radius=3, blur=5)
    heat.update([(500, 500), (-300, -300)])
    assert heat.accumulated.sum() == 0


def test_mask_buffer_is_cleared_between_updates():
    heat = HeatmapAccumulator(40, 30, decay=0.0, radius=3, blur=5)
    heat.update([(10, 10)])
    heat.update([(30, 20)])
    assert heat.accumulated[10, 10] == 0  # decay 0: only the last frame remains
    assert heat.accumulated[20, 30] > 0


def test_stationary_person_converges_to_geometric_limit():
    heat = HeatmapAccumulator(40, 30, decay=0.9, intensity=2.0, radius=4, blur=5)
    single = cv2.GaussianBlur(
        cv2.circle(np.zeros((30, 40), np.float32), (20, 15), 4, 2.0, -1), (5, 5), 0)
    for _ in range(300):
        heat.update([(20, 15)])
    assert np.allclose(heat.accumulated, single / (1 - 0.9), rtol=1e-3)


def test_normalized_is_uint8_and_empty_map_is_zero():
    heat = HeatmapAccumulator(20, 10)
    assert heat.normalized().dtype == np.uint8
    assert heat.normalized().max() == 0
    assert heat.colorized().shape == (10, 20, 3)


def test_normalized_peak_is_255_and_never_wraps():
    heat = HeatmapAccumulator(60, 40, decay=1.0, radius=3, blur=5)
    for _ in range(1000):
        heat.update([(30, 20)])
    norm = heat.normalized()
    assert norm.max() == 255
    assert norm[20, 30] == 255


def test_intensity_does_not_change_the_rendered_map():
    a = HeatmapAccumulator(60, 40, intensity=1.0, radius=3, blur=5)
    b = HeatmapAccumulator(60, 40, intensity=9.0, radius=3, blur=5)
    for pts in ([(10, 10)], [(40, 30)], [(10, 10)]):
        a.update(pts)
        b.update(pts)
    assert np.abs(a.normalized().astype(int) - b.normalized().astype(int)).max() <= 1


def test_blend_keeps_frame_shape_and_dtype():
    heat = HeatmapAccumulator(30, 20, radius=2, blur=3)
    heat.update([(10, 10)])
    frame = np.full((20, 30, 3), 200, np.uint8)
    out = heat.blend(frame, 0.6)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_blend_alpha_one_returns_frame():
    heat = HeatmapAccumulator(30, 20, radius=2, blur=3)
    heat.update([(10, 10)])
    frame = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert np.array_equal(heat.blend(frame, 1.0), frame)


def test_single_pixel_frame_does_not_crash():
    heat = HeatmapAccumulator(1, 1, radius=1, blur=3)
    heat.update([(0, 0)])
    assert heat.normalized().shape == (1, 1)


@pytest.mark.parametrize("kwargs", [
    {"blur": 4}, {"blur": 0}, {"decay": 1.5}, {"decay": -0.1}, {"radius": -1},
    {"intensity": float("nan")}, {"intensity": float("inf")},
    {"width": 0}, {"height": 0},
])
def test_invalid_arguments(kwargs):
    args = {"width": 10, "height": 10, **kwargs}
    with pytest.raises(ValueError):
        HeatmapAccumulator(**args)


def _half_life_seconds(decay, fps):
    """Seconds until a single event decays to half its weight."""
    heat = HeatmapAccumulator(10, 10, decay=decay, radius=1, blur=3, fps=fps)
    heat.update([(5, 5)])
    start = float(heat.accumulated.max())
    frames = 0
    while heat.accumulated.max() > start / 2:
        heat.update([])
        frames += 1
    return frames / fps


@pytest.mark.parametrize("fps", [15.0, 60.0])
def test_half_life_in_seconds_is_independent_of_fps(fps):
    assert _half_life_seconds(0.995, fps) == pytest.approx(_half_life_seconds(0.995, 30.0), abs=1 / 15)


def test_30_fps_decay_is_unchanged():
    assert effective_decay(0.995, 30.0) == 0.995
    heat = HeatmapAccumulator(40, 30, decay=0.9, radius=3, blur=5)
    default = HeatmapAccumulator(40, 30, decay=0.9, radius=3, blur=5, fps=30.0)
    for h in (heat, default):
        h.update([(20, 15)])
        h.update([])
    assert np.array_equal(heat.accumulated, default.accumulated)
    assert heat.frame_decay == 0.9


def test_effective_decay_at_15_fps_is_two_30_fps_steps():
    assert effective_decay(0.8, 15.0) == pytest.approx(0.64)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_is_rejected(fps):
    with pytest.raises(ValueError):
        effective_decay(0.9, fps)
    with pytest.raises(ValueError):
        HeatmapAccumulator(10, 10, fps=fps)
