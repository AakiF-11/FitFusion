"""
src/masking/human_parsing.py
Wraps SCHP (humanparsing) to produce pixel-level segmentation labels.
"""
import sys
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON" / "preprocess" / "humanparsing"))


def run_human_parsing(image_path: str, ckpt_dir: str) -> np.ndarray:
    """
    Run SCHP human parsing on the person image.

    Args:
        image_path: Path to the person image.
        ckpt_dir:   Path to ckpt/humanparsing/ (parsing_atr.onnx, parsing_lip.onnx).

    Returns:
        schp_mask: (H, W) numpy array of integer class labels per pixel.
    """
    # TODO: integrate IDM-VTON's humanparsing wrapper
    # from parsing_api import OnnxModel
    # model = OnnxModel(ckpt_dir)
    # return model.predict(image_path)
    raise NotImplementedError("Human parsing integration — implement in next sprint.")
