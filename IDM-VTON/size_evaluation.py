"""
FitFusion — Size Evaluation Metrics (V3)
==========================================
Inspired by SV-VTON's size evaluation module, this computes quantitative
metrics to assess whether generated try-on images correctly reflect
garment size differences.

Key metrics:
  - Garment Area Ratio (GAR): ratio of garment pixels in generated vs reference
  - Size Consistency Score (SCS): whether size order is preserved across sizes
  - Width Delta: difference in garment width at key body positions
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import json
import os


# International sizing standards (approximate garment width in cm at bust)
SIZE_STANDARDS = {
    "4XS": {"bust_width": 76},
    "3XS": {"bust_width": 80},
    "2XS": {"bust_width": 84},
    "XS":  {"bust_width": 88},
    "S":   {"bust_width": 92},
    "M":   {"bust_width": 96},
    "L":   {"bust_width": 100},
    "XL":  {"bust_width": 104},
    "2XL": {"bust_width": 110},
    "3XL": {"bust_width": 116},
    "4XL": {"bust_width": 122},
}


def compute_garment_area_ratio(
    generated_img: np.ndarray,
    reference_img: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute the ratio of garment area in generated image vs reference.
    
    A value > 1.0 means the generated garment appears larger than reference.
    A value < 1.0 means smaller.
    """
    # Simple approach: count non-background pixels in the masked region
    gen_masked = generated_img * (mask > 128)[:, :, None]
    ref_masked = reference_img * (mask > 128)[:, :, None]
    
    gen_nonzero = np.count_nonzero(gen_masked.sum(axis=2) > 30)
    ref_nonzero = np.count_nonzero(ref_masked.sum(axis=2) > 30)
    
    if ref_nonzero == 0:
        return 1.0
    return gen_nonzero / ref_nonzero


def measure_garment_width(image: np.ndarray, mask: np.ndarray, y_ratio: float = 0.4) -> float:
    """
    Measure the garment width at a specific vertical position.
    
    Args:
        image: (H, W, 3) image
        mask: (H, W) binary mask
        y_ratio: vertical position (0=top, 1=bottom), 0.4 ≈ bust level
    
    Returns:
        width in pixels
    """
    H, W = mask.shape
    y_pos = int(H * y_ratio)
    
    # Look at a band of rows around y_pos
    band = mask[max(0, y_pos-10):min(H, y_pos+10), :]
    if not band.any():
        return 0.0
    
    col_sum = band.sum(axis=0) > 0
    xs = np.where(col_sum)[0]
    if len(xs) < 2:
        return 0.0
    
    return float(xs[-1] - xs[0])


def compute_size_consistency_score(
    widths: dict,
) -> float:
    """
    Check if garment widths follow correct size ordering.
    
    If we generate S, M, L versions, width(S) < width(M) < width(L) should hold.
    Returns the fraction of correctly ordered pairs.
    """
    sizes_ordered = ["4XS", "3XS", "2XS", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL"]
    available = [(s, widths[s]) for s in sizes_ordered if s in widths and widths[s] > 0]
    
    if len(available) < 2:
        return 1.0
    
    correct = 0
    total = 0
    for i in range(len(available) - 1):
        for j in range(i + 1, len(available)):
            total += 1
            if available[j][1] >= available[i][1]:
                correct += 1
    
    return correct / total


def evaluate_size_accuracy(
    generated_images: dict,
    ground_truth_images: dict = None,
    masks: dict = None,
) -> dict:
    """
    Full size evaluation pipeline.
    
    Args:
        generated_images: {size_label: np.ndarray} mapping
        ground_truth_images: {size_label: np.ndarray} mapping (optional)
        masks: {size_label: np.ndarray} mapping (optional)
    
    Returns:
        metrics dict with GAR, SCS, width measurements, and SSIM
    """
    metrics = {}
    widths = {}
    
    for size_label, gen_img in generated_images.items():
        # Measure garment width at bust level
        if masks and size_label in masks:
            w = measure_garment_width(gen_img, masks[size_label], y_ratio=0.4)
        else:
            # Use simple thresholding to estimate garment region
            gray = np.mean(gen_img, axis=2)
            mask_est = (gray > 30).astype(np.uint8) * 255
            w = measure_garment_width(gen_img, mask_est, y_ratio=0.4)
        
        widths[size_label] = w
        
        # Compute SSIM against ground truth if available
        if ground_truth_images and size_label in ground_truth_images:
            gt = ground_truth_images[size_label]
            score = ssim(gt, gen_img, channel_axis=2, data_range=255)
            metrics[f"ssim_{size_label}"] = float(score)
    
    # Size consistency
    scs = compute_size_consistency_score(widths)
    
    # Width deltas vs sizing standards
    width_deltas = {}
    if len(widths) >= 2:
        # Normalize widths to real-world cm using known standards
        reference_size = "M" if "M" in widths else list(widths.keys())[0]
        ref_width_px = widths.get(reference_size, 1)
        ref_width_cm = SIZE_STANDARDS.get(reference_size, {}).get("bust_width", 96)
        px_to_cm = ref_width_cm / max(ref_width_px, 1)
        
        for size_label, w_px in widths.items():
            estimated_cm = w_px * px_to_cm
            expected_cm = SIZE_STANDARDS.get(size_label, {}).get("bust_width", 96)
            width_deltas[size_label] = {
                "estimated_cm": round(estimated_cm, 1),
                "expected_cm": expected_cm,
                "error_cm": round(abs(estimated_cm - expected_cm), 1),
            }
    
    return {
        "size_consistency_score": float(scs),
        "garment_widths_px": widths,
        "width_deltas_vs_standard": width_deltas,
        **metrics,
    }


def log_size_metrics(metrics: dict, output_dir: str, step: int):
    """Append size evaluation metrics to the training log."""
    metrics_path = os.path.join(output_dir, "size_evaluation_metrics.json")
    all_metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    
    metrics["step"] = step
    all_metrics.append(metrics)
    
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
