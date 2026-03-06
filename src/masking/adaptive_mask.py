"""
src/masking/adaptive_mask.py
Wraps IDM-VTON/size_adaptive_mask.py —
scales agnostic mask geometry according to the target size ratio.
"""
import sys
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))


def generate_adaptive_mask(
    schp_mask: np.ndarray,
    garment_type: str,
    person_size: str,
    target_size: str,
    output_dir: Optional[str] = None,
) -> Image.Image:
    """
    Generate a size-adaptive agnostic mask by scaling mask geometry
    based on the ratio between target_size and person_size.

    Args:
        schp_mask:    (H, W) pixel-label array from human parsing.
        garment_type: "top", "pants", "dress", etc.
        person_size:  Customer's actual size (e.g. "M").
        target_size:  Garment's labeled size (e.g. "XL").
        output_dir:   If set, saves debug mask image here.

    Returns:
        agnostic_mask: PIL Image binary mask (white = inpaint region).
    """
    from size_adaptive_mask import SizeAdaptiveMask

    generator = SizeAdaptiveMask()
    return generator.generate(
        schp_mask=schp_mask,
        garment_type=garment_type,
        person_size=person_size,
        garment_size=target_size,
        output_dir=output_dir,
    )
