"""
src/preprocessing/background.py
Thin wrapper around fitfusion/utils/preprocessing.py
Adds save_dir support for intermediate output.
"""
import os
import sys
from pathlib import Path

# Resolve project root so imports work from both local Windows and RunPod Linux
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "IDM-VTON"))

from fitfusion.utils.preprocessing import standardize_background as _standardize


def standardize_background(image_path: str, save_dir: str = None) -> str:
    """
    Strip background from customer photo and replace with studio gray (238,238,238).
    Returns path to cleaned image.

    Args:
        image_path: Absolute path to input image.
        save_dir:   If set, saves the cleaned image here. Otherwise saves next to input.
    """
    clean_path = _standardize(image_path)

    if save_dir:
        import shutil
        dest = os.path.join(save_dir, os.path.basename(clean_path))
        shutil.move(clean_path, dest)
        return dest

    return clean_path
