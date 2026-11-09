"""Command-line interface for ``crowd-heatmap``."""

import argparse
import logging
from typing import List, Optional

from crowd_heatmap import __version__
from crowd_heatmap.config import Config
from crowd_heatmap.dialogs import default_output_path

logger = logging.getLogger("crowd_heatmap")


def _odd_positive(value: str) -> int:
    """Argparse type for a positive odd integer (Gaussian kernel size)."""
    number = int(value)
    if number < 1 or number % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser with defaults taken from :class:`Config`."""
    d = Config()
    p = argparse.ArgumentParser(
        prog="crowd-heatmap",
        description="Track people with YOLO + ByteTrack and render a side-by-side "
        "tracking | spatial heatmap video. Without --input, file dialogs are opened.",
    )
    p.add_argument("--input", "-i", help="Input video. If omitted, file dialogs open.")
    p.add_argument("--output", "-o",
                   help="Output video (default: <name>_proc<ext> next to the input).")
    p.add_argument("--model", default=d.model,
                   help="Ultralytics weights path or official name (default: %(default)s).")
    p.add_argument("--conf", type=float, default=d.conf,
                   help="Detection confidence threshold (default: %(default)s).")
    p.add_argument("--imgsz", type=int, default=d.imgsz,
                   help="Inference image size (default: %(default)s).")
    p.add_argument("--tracker", default=d.tracker,
                   help="Ultralytics tracker config (default: %(default)s).")
    p.add_argument("--decay", type=float, default=d.decay,
                   help="Per-frame heat decay factor (default: %(default)s).")
    p.add_argument("--intensity", type=float, default=d.intensity,
                   help="Heat added inside each foot-point circle (default: %(default)s).")
    p.add_argument("--radius", type=int, default=d.radius,
                   help="Foot-point circle radius in pixels (default: %(default)s).")
    p.add_argument("--blur", type=_odd_positive, default=d.blur,
                   help="Gaussian blur kernel size, odd (default: %(default)s).")
    p.add_argument("--alpha", type=float, default=d.alpha,
                   help="Weight of the original frame in the heatmap blend "
                   "(default: %(default)s).")
    p.add_argument("--trail", type=int, default=d.trail,
                   help="Trajectory length in points per track (default: %(default)s).")
    p.add_argument("--save-heatmap", metavar="PATH",
                   help="Also save the final accumulated heatmap as an image (e.g. .png).")
    p.add_argument("--no-display", action="store_true",
                   help="Do not open the live preview window.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Parses arguments, resolves paths and runs the pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    if args.trail < 1:
        logger.error("--trail must be >= 1")
        return 2

    input_path, output_path = args.input, args.output
    if not input_path:
        from crowd_heatmap.dialogs import select_input, select_output

        input_path = select_input()
        if not input_path:
            logger.info("No input file selected. Exiting.")
            return 0
        output_path = output_path or select_output(input_path)
        if not output_path:
            logger.info("No output file selected. Exiting.")
            return 0
    output_path = output_path or default_output_path(input_path)
    cfg = Config(model=args.model, conf=args.conf, imgsz=args.imgsz, tracker=args.tracker,
                 decay=args.decay, intensity=args.intensity, radius=args.radius,
                 blur=args.blur, alpha=args.alpha, trail=args.trail,
                 display=not args.no_display, save_heatmap=args.save_heatmap)

    from crowd_heatmap.pipeline import process_video

    try:
        process_video(input_path, output_path, cfg)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
