"""Tests for argument parsing and validation in the CLI."""

import pytest

from crowd_heatmap.cli import build_parser, main
from crowd_heatmap.config import Config
from crowd_heatmap.dialogs import default_output_path


def test_parser_defaults_come_from_config():
    args = build_parser().parse_args([])
    d = Config()
    assert (args.conf, args.decay, args.blur, args.radius, args.trail) == (
        d.conf, d.decay, d.blur, d.radius, d.trail)


@pytest.mark.parametrize("flag", [
    ["--trail", "0"], ["--conf", "1.5"], ["--conf", "nan"], ["--imgsz", "0"],
    ["--decay", "-0.1"], ["--decay", "nan"], ["--intensity", "0"],
    ["--intensity", "nan"], ["--intensity", "inf"], ["--radius", "-1"],
    ["--alpha", "2"],
])
def test_out_of_range_values_return_exit_code_2(flag):
    assert main(["--input", "x.mp4"] + flag) == 2


@pytest.mark.parametrize("blur", ["4", "0", "-3", "abc"])
def test_bad_blur_is_rejected_by_argparse(blur):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--blur", blur])
    assert exc.value.code == 2


def test_output_equal_to_input_returns_exit_code_1(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    assert main(["-i", str(video), "-o", str(video), "--no-display"]) == 1
    assert video.read_bytes() == b"x"


def test_default_output_path():
    assert default_output_path("dir/clip.mp4") == "dir/clip_proc.mp4"
    assert default_output_path("clip") == "clip_proc"
