"""
Generate DensePose-like IUV body-part maps using a human parsing model.

Since Detectron2 cannot be installed (CUDA toolkit mismatch), we use
a HuggingFace human parsing model (segformer_b2_clothes) to create
body-region maps that approximate the DensePose IUV format.

The IDM-VTON UNet uses these maps for body-part spatial awareness when
determining garment placement and deformation.

IUV format:
  Channel 0 (I) = body part index (0-24)
  Channel 1 (U) = U surface coordinate (0-255)
  Channel 2 (V) = V surface coordinate (0-255)

Usage:
  python generate_densepose.py --data_dir data/fitfusion_vitonhd
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch


# ── DensePose body part mapping ──
# segformer_b2_clothes labels → DensePose-like I values
# segformer labels: Background, Hat, Hair, Sunglasses, Upper-clothes,
# Skirt, Pants, Dress, Belt, Left-shoe, Right-shoe, Face, Left-leg,
# Right-leg, Left-arm, Right-arm, Bag, Scarf
SEGFORMER_TO_DENSEPOSE = {
    0: 0,     # Background → 0
    1: 23,    # Hat → head (part 23)
    2: 23,    # Hair → head
    3: 23,    # Sunglasses → head
    4: 2,     # Upper-clothes → torso front (part 2)
    5: 2,     # Skirt → torso/lower
    6: 10,    # Pants → upper leg region
    7: 2,     # Dress → torso
    8: 2,     # Belt → torso
    9: 14,    # Left-shoe → left foot (part 14)
    10: 15,   # Right-shoe → right foot (part 15)
    11: 23,   # Face → head
    12: 11,   # Left-leg → left upper leg (part 11)
    13: 10,   # Right-leg → right upper leg (part 10)
    14: 5,    # Left-arm → left upper arm (part 5)
    15: 4,    # Right-arm → right upper arm (part 4)
    16: 0,    # Bag → background
    17: 24,   # Scarf → neck area
}

# Color palette for DensePose visualization (24 body parts)
DENSEPOSE_COLORS = np.array([
    [0, 0, 0],        # 0: background
    [30, 144, 255],   # 1: torso (right)
    [0, 128, 0],      # 2: torso front
    [0, 100, 0],      # 3: torso back
    [255, 69, 0],     # 4: right upper arm
    [255, 140, 0],    # 5: left upper arm
    [255, 165, 0],    # 6: right lower arm
    [255, 200, 0],    # 7: left lower arm
    [255, 105, 180],  # 8: right hand
    [255, 20, 147],   # 9: left hand
    [138, 43, 226],   # 10: right upper leg
    [148, 0, 211],    # 11: left upper leg
    [75, 0, 130],     # 12: right lower leg
    [123, 104, 238],  # 13: left lower leg
    [0, 255, 255],    # 14: right foot
    [0, 206, 209],    # 15: left foot
    [128, 0, 0],      # 16: inner right
    [192, 0, 0],      # 17: inner left
    [64, 128, 128],   # 18: outer right
    [0, 128, 128],    # 19: outer left
    [255, 255, 0],    # 20: right calf
    [255, 215, 0],    # 21: left calf
    [173, 216, 230],  # 22: right thigh
    [135, 206, 235],  # 23: head
    [70, 130, 180],   # 24: neck
], dtype=np.uint8)


def generate_iuv_from_segmentation(seg_map: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Convert segmentation labels to DensePose-like IUV map.
    
    Args:
        seg_map: (H, W) array of segformer label indices
        height: target height
        width: target width
    
    Returns:
        iuv: (H, W, 3) uint8 array in IUV format
    """
    iuv = np.zeros((height, width, 3), dtype=np.uint8)
    
    for seg_label, dp_part in SEGFORMER_TO_DENSEPOSE.items():
        if dp_part == 0:
            continue
        
        mask = (seg_map == seg_label)
        if not mask.any():
            continue
        
        # I channel: body part index (scaled for visibility)
        iuv[mask, 0] = min(dp_part * 10, 255)
        
        # U/V channels: normalized position within each body part region
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            x_range = max(x_max - x_min, 1)
            y_range = max(y_max - y_min, 1)
            iuv[ys, xs, 1] = ((xs - x_min) * 255 / x_range).astype(np.uint8)
            iuv[ys, xs, 2] = ((ys - y_min) * 255 / y_range).astype(np.uint8)
    
    return iuv


def main():
    parser = argparse.ArgumentParser(description="Generate DensePose-like IUV maps")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load segmentation model
    print("Loading human parsing model...")
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    ).to(device).eval()
    
    print(f"Model on {device}")
    
    for phase in ["train", "test"]:
        image_dir = data_dir / phase / "image"
        densepose_dir = data_dir / phase / "image-densepose"
        
        if not image_dir.exists():
            print(f"  [SKIP] {phase}/image not found")
            continue
        
        densepose_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        print(f"\n  Processing {len(images)} {phase} images...")
        
        # Process in batches
        for i in tqdm(range(0, len(images), args.batch_size), desc=f"  {phase} DensePose"):
            batch_paths = images[i:i + args.batch_size]
            batch_images = []
            valid_paths = []
            
            for img_path in batch_paths:
                out_path = densepose_dir / img_path.name
                # Skip existing files (allows resume)
                if out_path.exists():
                    # Check if existing file is blank (all zeros)
                    existing = np.array(Image.open(out_path))
                    if existing.max() > 0:
                        continue  # Already has real data
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"    [ERR] {img_path.name}: {e}")
            
            if not batch_images:
                continue
            
            # Run segmentation
            with torch.no_grad():
                inputs = processor(images=batch_images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits  # (B, num_labels, H/4, W/4)
            
            # Process each image in batch
            for j, (img, img_path) in enumerate(zip(batch_images, valid_paths)):
                w, h = img.size
                
                # Upsample logits to original size
                upsampled = torch.nn.functional.interpolate(
                    logits[j:j+1],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
                
                # Convert to IUV
                iuv = generate_iuv_from_segmentation(seg_map, h, w)
                
                # Save
                out_path = densepose_dir / img_path.name
                Image.fromarray(iuv).save(str(out_path))
    
    # Verify
    for phase in ["train", "test"]:
        dp_dir = data_dir / phase / "image-densepose"
        img_dir = data_dir / phase / "image"
        if dp_dir.exists() and img_dir.exists():
            n_dp = len(list(dp_dir.glob("*.jpg")) + list(dp_dir.glob("*.png")))
            n_img = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            non_blank = 0
            for f in dp_dir.iterdir():
                arr = np.array(Image.open(f))
                if arr.max() > 0:
                    non_blank += 1
            print(f"\n  {phase}: {n_dp}/{n_img} DensePose maps, {non_blank} non-blank ({non_blank/max(n_dp,1)*100:.0f}%)")
    
    print("\n✓ DensePose generation complete!")


if __name__ == "__main__":
    main()
