"""Tkinter helpers: file dialogs and screen resolution."""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]


def default_output_path(input_path: str) -> str:
    """Returns ``<name>_proc<ext>`` next to the input file.

    Args:
        input_path: Path of the input video.

    Returns:
        The default output path.
    """
    base, ext = os.path.splitext(input_path)
    return f"{base}_proc{ext}"


def get_screen_resolution() -> Optional[Tuple[int, int]]:
    """Returns the primary screen size, or ``None`` if no display is available."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception as exc:  # TclError when headless, ImportError without Tk.
        logger.warning("Could not read screen resolution: %s", exc)
        return None


def select_input() -> Optional[str]:
    """Opens a dialog to choose the input video.

    Returns:
        The selected path, or ``None`` if the dialog was cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select input video", filetypes=VIDEO_TYPES)
    root.destroy()
    return path or None


def select_output(input_path: str) -> Optional[str]:
    """Opens a save dialog pre-filled with ``<name>_proc<ext>``.

    Args:
        input_path: Selected input video, used for the default name and extension.

    Returns:
        The chosen output path, or ``None`` if the dialog was cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    ext = os.path.splitext(input_path)[1]
    default = default_output_path(input_path)
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        title="Save processed video as",
        initialdir=os.path.dirname(default) or None,
        initialfile=os.path.basename(default),
        defaultextension=ext,
        filetypes=[("Video files", f"*{ext}"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None
