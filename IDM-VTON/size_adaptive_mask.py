"""
FitFusion — Size-Adaptive Mask Generation (V3)
================================================
Inspired by SV-VTON (arXiv:2504.00562), this module generates
size-specific agnostic masks that adapt based on the garment size
relative to the person's body size.

Key insight: When trying on an XL shirt on a S person, the agnostic
mask should be WIDER than the body contour to accommodate the excess
fabric. Conversely, a S shirt on an XL person should have a TIGHTER
mask that follows body contours closely.

Usage:
    python size_adaptive_mask.py --data_dir data/fitfusion_vitonhd
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import json


def compute_size_adaptive_mask(
    body_mask: np.ndarray,
    person_size_idx: int,
    garment_size_idx: int,
    garment_type: str = "top",
    max_dilation: int = 40,
    max_erosion: int = 15,
) -> np.ndarray:
    """
    Generate a size-adaptive agnostic mask based on person↔garment size gap.
    
    When the garment is LARGER than the person → dilate mask (more room for excess fabric)
    When the garment is SMALLER than the person → erode mask slightly (tighter fit)
    When sizes match → use original mask
    
    Args:
        body_mask: (H, W) binary mask of the clothing region (0 or 255)
        person_size_idx: person's size index (0=4XS, 5=M, 10=4XL)
        garment_size_idx: garment's size index
        garment_type: type of garment (affects dilation direction)
        max_dilation: maximum dilation in pixels for extreme size gaps
        max_erosion: maximum erosion in pixels for tight fits
    
    Returns:
        adapted_mask: (H, W) uint8 mask adapted for the size relationship
    """
    size_gap = garment_size_idx - person_size_idx  # positive = garment larger
    
    if size_gap == 0:
        return body_mask
    
    # Determine kernel size based on size gap magnitude
    if size_gap > 0:
        # Garment is LARGER → DILATE mask to accommodate excess fabric
        # Larger gap = more dilation
        kernel_size = min(int(abs(size_gap) * 10), max_dilation)
        
        # Different dilation directions based on garment type
        if garment_type in ["top", "shirt", "t-shirt", "hoodie", "sweater", "jacket"]:
            # For tops: dilate more horizontally (width) than vertically
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size * 2, kernel_size)
            )
        elif garment_type in ["pants", "jeans", "trousers", "leggings", "tights"]:
            # For bottoms: dilate more vertically (length)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size * 2)
            )
        elif garment_type in ["dress"]:
            # Dresses: dilate uniformly
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, int(kernel_size * 1.5))
            )
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
        
        adapted_mask = cv2.dilate(body_mask, kernel, iterations=1)
        
    else:
        # Garment is SMALLER → ERODE mask slightly (tighter fit)
        kernel_size = min(int(abs(size_gap) * 5), max_erosion)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        adapted_mask = cv2.erode(body_mask, kernel, iterations=1)
    
    # Smooth the mask edges
    adapted_mask = cv2.GaussianBlur(adapted_mask, (5, 5), 2)
    adapted_mask = (adapted_mask > 128).astype(np.uint8) * 255
    
    return adapted_mask


SIZE_TO_INDEX = {
    "4XS": 0, "3XS": 1, "2XS": 2, "XS": 3, "S": 4,
    "M": 5, "L": 6, "XL": 7, "2XL": 8, "3XL": 9, "4XL": 10,
}


def generate_adaptive_masks(data_dir: str, measurements_path: str = None):
    """
    Generate size-adaptive masks for all training pairs.
    
    For each (person, garment) pair in the dataset, create a mask adapted
    to the size relationship between person and garment.
    """
    data_dir = Path(data_dir)
    
    # Load measurements
    measurements = {}
    for phase in ["train", "test"]:
        meas_file = data_dir / phase / "measurements.json"
        if meas_file.exists():
            with open(meas_file) as f:
                meas_data = json.load(f)
                measurements.update(meas_data)
    
    print(f"Loaded measurements for {len(measurements)} images")
    
    for phase in ["train", "test"]:
        mask_dir = data_dir / phase / "agnostic-mask"
        adaptive_mask_dir = data_dir / phase / "agnostic-mask-adaptive"
        pairs_file = data_dir / phase / "pairs.txt"
        
        if not mask_dir.exists() or not pairs_file.exists():
            print(f"  [SKIP] {phase}: missing mask dir or pairs file")
            continue
        
        adaptive_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Read pairs
        with open(pairs_file) as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        print(f"\n  Processing {len(pairs)} {phase} pairs...")
        
        for im_name, c_name in tqdm(pairs, desc=f"  {phase} adaptive masks"):
            # Load original mask
            mask_stem = os.path.splitext(im_name)[0]
            mask_name = mask_stem + "_mask.png"
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                continue
            
            mask = np.array(Image.open(mask_path).convert("L"))
            
            # Get size info
            person_m = measurements.get(im_name, {})
            garment_m = measurements.get(c_name, {})
            
            person_size = SIZE_TO_INDEX.get(
                person_m.get("size_label", "M").upper(), 5
            )
            garment_size = SIZE_TO_INDEX.get(
                garment_m.get("size_label", "M").upper(), 5
            )
            garment_type = person_m.get("garment_type", "top")
            
            # Generate adaptive mask
            adapted = compute_size_adaptive_mask(
                mask, person_size, garment_size, garment_type
            )
            
            # Save with pair-specific name
            out_name = f"{mask_stem}_{os.path.splitext(c_name)[0]}_mask.png"
            Image.fromarray(adapted).save(str(adaptive_mask_dir / out_name))
    
    print("\n✓ Adaptive mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    generate_adaptive_masks(args.data_dir)
