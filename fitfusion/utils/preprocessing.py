import os
import logging
from pathlib import Path
from PIL import Image
from typing import Any, Tuple
import numpy as np
import cv2

log = logging.getLogger(__name__)

# ── RMBG-1.4 (BRIA) background removal ────────────────────────────────────────
# RMBG-1.4 is a BiRefNet-based model that produces far cleaner person/garment
# masks than the older rembg u2net model. It is downloaded during pod setup to
# /workspace/model_cache/RMBG-1.4. Falls back to rembg if not available.

# Cache dir where runpod_setup.sh writes the model.
_RMBG_CACHE = os.environ.get("RMBG_MODEL_DIR", "/workspace/model_cache/RMBG-1.4")
_rmbg_pipe = None  # lazy-loaded singleton


def _load_rmbg_pipe():
    """Lazy-load the BRIA RMBG-1.4 pipeline (once per process)."""
    global _rmbg_pipe
    if _rmbg_pipe is not None:
        return _rmbg_pipe

    try:
        from transformers import pipeline as hf_pipeline
        import torch

        model_source = _RMBG_CACHE if os.path.isdir(_RMBG_CACHE) else "briaai/RMBG-1.4"
        device = 0 if torch.cuda.is_available() else -1  # GPU index or CPU (-1)

        log.info(f"Loading RMBG-1.4 from {model_source} (device={'cuda' if device==0 else 'cpu'})...")
        _rmbg_pipe = hf_pipeline(
            "image-segmentation",
            model=model_source,
            trust_remote_code=True,
            device=device,
        )
        log.info("RMBG-1.4 loaded.")
        return _rmbg_pipe
    except Exception as exc:
        log.warning(f"RMBG-1.4 unavailable ({exc}), will fall back to rembg.")
        return None


def _remove_bg_rmbg(img_rgba: Image.Image) -> Image.Image:
    """
    Use BRIA RMBG-1.4 to remove the background.
    Returns an RGBA image (background = fully transparent).
    """
    pipe = _load_rmbg_pipe()
    if pipe is None:
        raise RuntimeError("RMBG-1.4 not available")

    img_rgb = img_rgba.convert("RGB")
    result = pipe(img_rgb)
    # HF pipelines may return [{"score":..., "mask": PIL.Image}] OR a PIL.Image directly.
    raw = result[0] if isinstance(result, list) else result
    mask = raw["mask"] if isinstance(raw, dict) else raw  # "L" mode — white=fg, black=bg
    mask = mask.convert("L").resize(img_rgb.size, Image.LANCZOS)

    out = img_rgb.convert("RGBA")
    out.putalpha(mask)
    return out


def _remove_bg_rembg(img_rgba: Image.Image) -> Image.Image:
    """Fallback: use the rembg u2net model (always available)."""
    from rembg import remove
    return remove(img_rgba)


def standardize_background(input_image_path: str, bg_color: Tuple[int, int, int] = (238, 238, 238)) -> str:
    """
    Strips the background and replaces it with a solid studio gray (238, 238, 238).

    Uses BRIA RMBG-1.4 for best quality (downloaded at pod setup time).
    Falls back to rembg / u2net automatically if RMBG-1.4 isn't available.

    Returns the path to the cleaned image.
    """
    input_img = Image.open(input_image_path).convert("RGBA")

    # Try RMBG-1.4 first; fall back to rembg on any error
    try:
        no_bg_img = _remove_bg_rmbg(input_img)
        log.info("  Background removed with RMBG-1.4.")
    except Exception as exc:
        log.warning(f"  RMBG-1.4 failed ({exc}), using rembg fallback.")
        no_bg_img = _remove_bg_rembg(input_img)

    # Composite foreground over studio gray
    gray_bg = Image.new("RGBA", no_bg_img.size, (*bg_color, 255))
    clean_img = Image.alpha_composite(gray_bg, no_bg_img).convert("RGB")

    base, _ = os.path.splitext(input_image_path)
    output_path = f"{base}_studio_bg.png"
    clean_img.save(output_path)
    return output_path

def erase_neckline(person_image: Image.Image, openpose_keypoints: Any, neckline_type: str, color: Tuple[int, int, int] = (238, 238, 238)) -> Image.Image:
    """
    Erases underlying high-neck clothing to prepare for low-cut garments.
    Draws a polygon from neck and shoulders down to chest.
    """
    if not openpose_keypoints or not neckline_type:
        return person_image
        
    if neckline_type.lower() not in ["v-neck", "scoop", "deep-u"]:
        return person_image
        
    points = []
    if isinstance(openpose_keypoints, dict):
        points = list(openpose_keypoints.values())
    elif isinstance(openpose_keypoints, list) or isinstance(openpose_keypoints, np.ndarray):
        points = openpose_keypoints
        
    def get_pt(idx):
        if idx < len(points):
            p = points[idx]
            if p is not None and len(p) >= 2 and p[0] > 0 and p[1] > 0:
                return int(p[0]), int(p[1])
        return None
        
    neck = get_pt(1)
    r_should = get_pt(2)
    l_should = get_pt(5)
    
    if neck and r_should and l_should:
        img_np = np.array(person_image)
        
        # Chest drops proportionally to shoulder width
        shoulder_width = abs(l_should[0] - r_should[0])
        chest_y = int(neck[1] + shoulder_width * 0.45)
        
        # Polygon: Neck -> Right Shoulder -> Chest Center -> Left Shoulder
        pts = np.array([
            [neck[0], neck[1]],
            [min(neck[0], l_should[0] + int(shoulder_width * 0.2)), l_should[1]],
            [neck[0], chest_y],
            [max(neck[0], r_should[0] - int(shoulder_width * 0.2)), r_should[1]],
        ], np.int32)
        
        # Better: use proper ordering. 
        pts = np.array([
            neck,
            r_should,
            [neck[0], chest_y],
            l_should,
        ], np.int32)
        
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img_np, [pts], color)
        
        return Image.fromarray(img_np)
        
    return person_image

def desaturate_source_garment(original_image: Image.Image, schp_mask: np.ndarray) -> Image.Image:
    """
    Converts the original upper_clothes pixels to grayscale before inference
    to kill the chroma while maintaining the luma, preventing latent color bleeding.
    """
    if schp_mask is None:
        return original_image
        
    img_np = np.array(original_image.convert("RGB"))
    
    # Resize SCHP mask if it doesn't match the image dimensions
    if schp_mask.shape[:2] != img_np.shape[:2]:
        schp_mask_resized = cv2.resize(schp_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        schp_mask_resized = schp_mask
        
    # Standard SCHP Lip upper_clothes label is 4 or 5 depending on the dataset/model,
    # but IDM-VTON generally uses 4 for upper_clothes.
    # We will use 4, since it was used in size_aware_vton.py earlier (np.sum(schp_mask == 4))
    upper_clothes_mask = (schp_mask_resized == 4).astype(np.uint8)
    
    # Convert image to grayscale (luma only), then back to 3-channel
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_img_3d = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    # Create 3-channel boolean mask
    mask_3d = np.stack([upper_clothes_mask] * 3, axis=2)
    
    # Desaturate only where mask is True
    desaturated_np = np.where(mask_3d == 1, gray_img_3d, img_np)
    
    return Image.fromarray(desaturated_np.astype(np.uint8))
