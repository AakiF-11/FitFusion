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
    from size_adaptive_mask import compute_size_adaptive_mask, SIZE_TO_INDEX

    person_size_idx  = SIZE_TO_INDEX.get(person_size.upper(), 5)
    garment_size_idx = SIZE_TO_INDEX.get(target_size.upper(), 7)

    # Extract clothing pixels from human parsing label map
    # SCHP labels: 1-7 upper body, 8-12 lower body, 16=dress, 17=coat
    clothing_labels = {1, 2, 3, 4, 5, 6, 7, 16, 17}
    body_mask = np.zeros(schp_mask.shape[:2], dtype=np.uint8)
    for lbl in clothing_labels:
        body_mask[schp_mask == lbl] = 255

    adapted = compute_size_adaptive_mask(
        body_mask=body_mask,
        person_size_idx=person_size_idx,
        garment_size_idx=garment_size_idx,
        garment_type=garment_type,
    )

    result = Image.fromarray(adapted)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result.save(str(Path(output_dir) / "agnostic_mask_adaptive.png"))

    return result
