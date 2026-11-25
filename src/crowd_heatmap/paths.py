"""Path safety checks (standard library only)."""

import os
from typing import Optional


def _same_file(a: str, b: str) -> bool:
    """Tells whether two paths point to the same file (case and link aware)."""
    if os.path.exists(a) and os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def check_distinct_paths(
    input_path: str, output_path: str, heatmap_path: Optional[str] = None
) -> None:
    """Refuses output paths that would overwrite the input or each other.

    Writing the output video over the input while it is still being read would
    destroy the source, so this is checked before anything is opened.

    Args:
        input_path: Source video.
        output_path: Destination video.
        heatmap_path: Optional destination of the final heatmap image.

    Raises:
        ValueError: If the output equals the input, or the heatmap image path
            equals the input or the output video.
    """
    if _same_file(input_path, output_path):
        raise ValueError(f"Output must differ from the input video: {output_path}")
    if heatmap_path:
        if _same_file(input_path, heatmap_path):
            raise ValueError(f"Heatmap image must differ from the input video: {heatmap_path}")
        if _same_file(output_path, heatmap_path):
            raise ValueError(f"Heatmap image must differ from the output video: {heatmap_path}")
