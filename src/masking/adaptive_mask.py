"""
src/masking/adaptive_mask.py
Wraps IDM-VTON/size_adaptive_mask.py —
scales agnostic mask geometry according to the target size ratio.

ATR dataset label mapping (used by IDM-VTON humanparsing):
  0=Background, 1=Hat, 2=Hair, 3=Sunglasses,
  4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress,
  8=Belt, 9=Left shoe, 10=Right shoe, 11=Face,
  12=Left leg, 13=Right leg, 14=Left arm, 15=Right arm,
  16=Bag, 17=Scarf, 18=Neck (added from LIP by parsing_api)
"""
import sys
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
import cv2

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))

# ── ATR label constants ────────────────────────────────────────────────────────
_ATR_UPPER   = {4, 7, 17}          # Upper-clothes, Dress, Scarf
_ATR_LOWER   = {5, 6, 12, 13}     # Skirt, Pants, Left leg, Right leg
_ATR_ARMS    = {14, 15}            # Left arm, Right arm
_ATR_HEAD    = {1, 2, 3, 11, 18}  # Hat, Hair, Sunglasses, Face, Neck — NEVER inpaint here

# Per garment-type clothing regions to build the body_mask from
_CLOTHING_LABELS = {
    "top":       _ATR_UPPER | _ATR_ARMS,
    "shirt":     _ATR_UPPER | _ATR_ARMS,
    "t-shirt":   _ATR_UPPER | _ATR_ARMS,
    "hoodie":    _ATR_UPPER | _ATR_ARMS,
    "sweater":   _ATR_UPPER | _ATR_ARMS,
    "jacket":    _ATR_UPPER | _ATR_ARMS,
    "coat":      _ATR_UPPER | _ATR_ARMS,
    "outerwear": _ATR_UPPER | _ATR_ARMS,
    "blouse":    _ATR_UPPER | _ATR_ARMS,
    "pants":     _ATR_LOWER,
    "jeans":     _ATR_LOWER,
    "trousers":  _ATR_LOWER,
    "shorts":    _ATR_LOWER,
    "skirt":     {5},
    "tights":    _ATR_LOWER,
    "leggings":  _ATR_LOWER,
    "dress":     _ATR_UPPER | _ATR_LOWER | _ATR_ARMS,
    "bodysuit":  _ATR_UPPER | _ATR_LOWER | _ATR_ARMS,
}


def generate_adaptive_mask(
    schp_mask: np.ndarray,
    garment_type: str,
    person_size: str,
    target_size: str,
    output_dir: Optional[str] = None,
) -> Image.Image:
    """
    Generate a size-adaptive agnostic mask, scaled by the size ratio between
    target_size and person_size, with a hard face/head protection zone.

    Args:
        schp_mask:    (H, W) pixel-label array from IDM-VTON human parsing (ATR scheme).
        garment_type: "top", "jacket", "tights", "dress", etc.
        person_size:  Customer's actual size (e.g. "M").
        target_size:  Garment's labeled size (e.g. "XL").
        output_dir:   If set, saves debug masks here.

    Returns:
        agnostic_mask: PIL Image binary mask (white = inpaint region).
    """
    from size_adaptive_mask import compute_size_adaptive_mask, SIZE_TO_INDEX

    person_size_idx  = SIZE_TO_INDEX.get(person_size.upper(), 5)
    garment_size_idx = SIZE_TO_INDEX.get(target_size.upper(), 7)

    # ── 1. Build clothing body_mask using correct ATR labels ──────────────────
    gtype = garment_type.lower()
    clothing_labels = _CLOTHING_LABELS.get(gtype, _ATR_UPPER | _ATR_ARMS)

    body_mask = np.zeros(schp_mask.shape[:2], dtype=np.uint8)
    for lbl in clothing_labels:
        body_mask[schp_mask == lbl] = 255

    # ── 2. Compute size-adaptive dilation / erosion ───────────────────────────
    adapted = compute_size_adaptive_mask(
        body_mask=body_mask,
        person_size_idx=person_size_idx,
        garment_size_idx=garment_size_idx,
        garment_type=gtype,
    )

    # ── 3. HEAD PROTECTION — never let the mask touch the face/hair/neck ─────
    # Build a head zone from ATR head labels, then dilate it by 20px for safety.
    head_zone = np.zeros(schp_mask.shape[:2], dtype=np.uint8)
    for lbl in _ATR_HEAD:
        head_zone[schp_mask == lbl] = 255

    if head_zone.any():
        # Expand the head zone slightly so the mask edge doesn't graze the chin
        head_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        head_zone = cv2.dilate(head_zone, head_kernel, iterations=1)

    # Zero out any mask pixels that overlap with the protected head zone
    adapted[head_zone > 0] = 0

    # ── 4. Optionally save debug images ──────────────────────────────────────
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(body_mask).save(str(Path(output_dir) / "body_mask_raw.png"))
        Image.fromarray(adapted).save(str(Path(output_dir) / "agnostic_mask_adaptive.png"))
        Image.fromarray(head_zone).save(str(Path(output_dir) / "head_protection_zone.png"))

    return Image.fromarray(adapted)

