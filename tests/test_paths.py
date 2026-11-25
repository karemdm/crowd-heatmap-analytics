"""Tests for the output-path safety checks."""

import pytest

from crowd_heatmap.paths import check_distinct_paths


def test_distinct_paths_pass(tmp_path):
    check_distinct_paths(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"),
                         str(tmp_path / "h.png"))


def test_output_equal_to_input_is_refused(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError):
        check_distinct_paths(str(video), str(video))


def test_same_file_through_different_spelling_is_refused(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    alias = str(tmp_path / "sub" / ".." / "a.mp4")
    with pytest.raises(ValueError):
        check_distinct_paths(str(video), alias)


def test_same_missing_path_is_refused(tmp_path):
    with pytest.raises(ValueError):
        check_distinct_paths(str(tmp_path / "a.mp4"), str(tmp_path / "a.mp4"))


def test_heatmap_must_not_overwrite_input_or_output(tmp_path):
    with pytest.raises(ValueError):
        check_distinct_paths(str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"),
                             str(tmp_path / "in.mp4"))
    with pytest.raises(ValueError):
        check_distinct_paths(str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"),
                             str(tmp_path / "out.mp4"))
