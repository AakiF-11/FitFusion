"""
src/pose/densepose.py
Wraps IDM-VTON/generate_densepose.py to produce UV surface maps.
"""
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))


def generate_densepose_map(
    image_path: str,
    ckpt_dir: str,
    output_dir: Optional[str] = None,
):
    """
    Generate DensePose UV map for the person image.

    Args:
        image_path: Path to the person image.
        ckpt_dir:   Path to ckpt/densepose/ (model_final_162be9.pkl).
        output_dir: If set, saves the densepose image here.

    Returns:
        densepose_image: PIL Image of the UV surface map.
    """
    # TODO: integrate IDM-VTON/generate_densepose.py
    # from generate_densepose import DensePosePredictor
    # predictor = DensePosePredictor(ckpt_dir)
    # return predictor.run(image_path, output_dir)
    raise NotImplementedError("DensePose integration — implement in next sprint.")
