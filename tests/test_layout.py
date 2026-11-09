"""Tests for the fit-to-screen scale helpers."""

from crowd_heatmap.layout import fit_scale, scaled_size


def test_large_frame_is_limited_by_width():
    # Side-by-side 1920x1080 -> 3840x1080 on a 1920x1080 screen.
    ratio = fit_scale(3840, 1080, 1920, 1080)
    assert abs(ratio - 1728 / 3840) < 1e-9


def test_tall_frame_is_limited_by_height():
    ratio = fit_scale(1000, 2000, 1920, 1080)
    assert abs(ratio - 918 / 2000) < 1e-9


def test_small_frame_is_never_upscaled():
    assert fit_scale(640, 360, 1920, 1080) == 1.0


def test_custom_fractions():
    assert fit_scale(2000, 1000, 2000, 1000, frac_w=0.5, frac_h=1.0) == 0.5


def test_invalid_frame_size():
    try:
        fit_scale(0, 100, 1920, 1080)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_scaled_size():
    assert scaled_size(3840, 1080, 0.45) == (1728, 486)
    assert scaled_size(10, 10, 0.0) == (1, 1)
