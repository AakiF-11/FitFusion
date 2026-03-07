"""
FitFusion — Definitive Size-Aware VTON Pipeline
=================================================
The complete physics-based pipeline that transforms IDM-VTON into a
size-aware virtual try-on system without any additional training.

7 Physics Layers:
    1. Mathematical garment resizing (width/length ratios)
    2. Physics-descriptive text prompts (loose/tight/drape descriptions)
    3. TPS warp intensity control (gentle for loose, aggressive for tight)
    4. DensePose-guided regional masking (shoulder, torso, arms separately)
    5. Gradient mask edges (soft edges for loose, hard for tight)
    6. Inpainting strength control (more creative freedom for larger gaps)
    7. Garment-type-specific behavior (shirts vs pants vs dresses)

Usage:
    from size_aware_vton import SizeAwareVTON
    
    engine = SizeAwareVTON()
    result = engine.generate(
        person_image="person.jpg",
        garment_image="garment.jpg",
        person_size="S",
        garment_size="XL",
        garment_type="top",
    )
"""

import numpy as np
from PIL import Image, ImageFilter
import cv2
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent to path if needed for fitfusion import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from size_charts import compute_size_ratio, get_garment_dimensions


# ════════════════════════════════════════════════════════════════
#  Layer 1: Fit Classification
# ════════════════════════════════════════════════════════════════

class FitType(Enum):
    VERY_TIGHT = -3    # 3+ sizes smaller
    TIGHT = -2         # 2 sizes smaller
    SNUG = -1          # 1 size smaller
    STANDARD = 0       # exact size
    RELAXED = 1        # 1 size larger
    LOOSE = 2          # 2 sizes larger
    OVERSIZED = 3      # 3+ sizes larger


@dataclass
class FitProfile:
    """Describes exactly how the garment should behave physically."""
    fit_type: FitType
    size_gap: int
    width_ratio: float
    length_ratio: float
    
    # Physics parameters (computed from size_gap)
    warp_intensity: float       # 0.0 (no warp) to 1.0 (aggressive body-hugging)
    mask_expansion_px: int      # pixels to dilate/erode mask
    inpainting_strength: float  # 0.0 to 1.0 — creative freedom for the model
    edge_softness: int          # gaussian blur kernel for mask edges
    fabric_tension: float       # 0.0 (loose/draped) to 1.0 (stretched/taut)
    
    # Prompt descriptors
    positive_prompt: str
    negative_prompt: str
    
    # Validation gates
    halt_generation: bool = False
    error_msg: str = ""


def classify_fit(
    garment_type: str,
    garment_size: str, 
    person_size: str,
    fabric_rigidity: float = 0.5,
) -> FitProfile:
    """
    Classify the fit relationship and compute ALL physics parameters.
    This is the brain of the entire system.
    """
    ratio = compute_size_ratio(garment_type, garment_size, person_size)
    gap = ratio["size_gap"]
    w_ratio = ratio["width_ratio"]
    l_ratio = ratio["length_ratio"]
    
    # [TASK 3 MOD: Fabric Rigidity Coefficients]
    # Ensure highly rigid fabrics (0.9) cannot dilate beyond 1.05,
    # while soft fabrics (0.1) can dilate up to 1.30 for natural pooling.
    max_limit = float(np.interp(fabric_rigidity, [0.1, 0.9], [1.30, 1.05]))
    if w_ratio > max_limit:
        w_ratio = max_limit
        
    # Clamp gap to enum range
    clamped_gap = max(-3, min(3, gap))
    fit_type = FitType(clamped_gap)
    
    # ── Physics parameter computation ──
    
    # Warp intensity: tight = aggressive (1.0), loose = gentle (0.3)
    # Standard fit = 0.7 (natural body-following)
    if gap <= -2:
        warp_intensity = 1.0     # Pull fabric tight to body
    elif gap == -1:
        warp_intensity = 0.85    # Snug but not stretched
    elif gap == 0:
        warp_intensity = 0.7     # Standard natural drape
    elif gap == 1:
        warp_intensity = 0.5     # Relaxed, some distance from body
    elif gap == 2:
        warp_intensity = 0.35    # Loose, fabric hangs away from body
    else:
        warp_intensity = 0.2     # Oversized, minimal body conformity
    
    # Mask expansion: positive = dilate (room for excess fabric)
    if gap > 0:
        mask_expansion_px = int(gap * 20)   # 20px per size up (stronger size differentiation)
    elif gap < 0:
        mask_expansion_px = int(gap * 8)    # 8px per size down
    else:
        mask_expansion_px = 0
    
    # Inpainting strength: larger gaps need more creative freedom
    # to render wrinkles, folds, stretching
    inpainting_strength = min(0.6 + abs(gap) * 0.08, 0.95)
    
    # Edge softness: loose garments have softer edges (fabric movement)
    # tight garments have sharp edges (fabric pressed against body)
    if gap >= 2:
        edge_softness = 11   # Very soft, fabric edges are diffuse
    elif gap >= 1:
        edge_softness = 7    # Moderately soft
    elif gap == 0:
        edge_softness = 5    # Standard
    elif gap >= -1:
        edge_softness = 3    # Sharper edges
    else:
        edge_softness = 1    # Very sharp, fabric clings to body contour
    
    # Fabric tension: 0 = loose/flowing, 1 = stretched/taut
    fabric_tension = max(0.0, min(1.0, 0.5 - gap * 0.15))
    
    # ── Prompt engineering ──
    pos_prompt, neg_prompt = _build_fit_prompts(fit_type, garment_type, gap)
    
    return FitProfile(
        fit_type=fit_type,
        size_gap=gap,
        width_ratio=w_ratio,
        length_ratio=l_ratio,
        warp_intensity=warp_intensity,
        mask_expansion_px=mask_expansion_px,
        inpainting_strength=inpainting_strength,
        edge_softness=edge_softness,
        fabric_tension=fabric_tension,
        positive_prompt=pos_prompt,
        negative_prompt=neg_prompt,
    )


# ════════════════════════════════════════════════════════════════
#  Layer 2: Physics-Descriptive Prompt Engineering
# ════════════════════════════════════════════════════════════════

def _build_fit_prompts(fit_type: FitType, garment_type: str, gap: int) -> tuple:
    """
    Build precise text prompts that include garment type and fit keywords.
    The garment type word anchors IDM-VTON to the correct garment category,
    and anti-dress negatives prevent full-body product shots from bleeding
    through as a long dress/gown on the output.
    """
    gtype = garment_type.lower()
    garment_word = {
        "top": "t-shirt", "shirt": "shirt", "tee": "t-shirt", "blouse": "blouse",
        "hoodie": "hoodie", "sweater": "sweater", "jacket": "jacket",
        "pants": "pants", "jeans": "jeans", "skirt": "skirt", "dress": "dress",
        "tights": "tights", "leggings": "leggings",
    }.get(gtype, gtype)

    fit_word = {
        FitType.VERY_TIGHT: "very tight body-hugging",
        FitType.TIGHT: "tight slim-fit",
        FitType.SNUG: "snug close-fitting",
        FitType.STANDARD: "standard relaxed",
        FitType.RELAXED: "relaxed comfortable",
        FitType.LOOSE: "loose",
        FitType.OVERSIZED: "very oversized baggy",
    }.get(fit_type, "")

    base = "high quality fashion photograph, professional studio lighting, sharp focus"
    pos = f"{base}, person wearing a {fit_word} {garment_word}, upper body visible"

    garment_neg = ""
    if gtype in ("top", "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket"):
        garment_neg = "full-length dress, long gown, long skirt, pants becoming dress, "
    neg = (
        f"monochrome, lowres, bad anatomy, worst quality, low quality, artifacts, "
        f"{garment_neg}naked, bare chest, topless, extra limbs"
    )
    return pos, neg


# ════════════════════════════════════════════════════════════════
#  Layer 3: Garment Resizer (Enhanced)
# ════════════════════════════════════════════════════════════════

def resize_garment(
    garment_image: Image.Image,
    fit_profile: FitProfile,
    garment_type: str,
) -> Image.Image:
    """
    Crop the garment image to the relevant body zone so IDM-VTON's garment
    encoder only sees the target garment — not pants/legs from full-body
    product shots (which would make the model generate a long dress).
    """
    w, h = garment_image.size
    gtype = garment_type.lower()

    if gtype in ("top", "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket"):
        # Keep only the upper 55%: captures torso/shirt, removes pants and shoes.
        # Resize back to original dimensions so the downstream 768x1024 scale
        # fills the canvas with a properly zoomed-in view of the garment.
        crop_h = int(h * 0.55)
        garment_image = garment_image.crop((0, 0, w, crop_h)).resize((w, h), Image.LANCZOS)
    elif gtype in ("pants", "jeans", "trousers", "skirt", "tights", "leggings"):
        # Keep only the lower 60%: captures pants/skirt, removes upper body.
        crop_y = int(h * 0.40)
        bottom = garment_image.crop((0, crop_y, w, h))
        result = Image.new("RGB", (w, h), (238, 238, 238))
        result.paste(bottom.resize((w, h - crop_y), Image.LANCZOS), (0, crop_y))
        garment_image = result
    # dress: no crop — encoder needs to see the full garment

    return garment_image



# ════════════════════════════════════════════════════════════════
#  Layer 4: DensePose-Guided Regional Masking
# ════════════════════════════════════════════════════════════════

def create_regional_mask(
    person_image: Image.Image,
    densepose_map: Optional[np.ndarray],
    fit_profile: FitProfile,
    garment_type: str,
    schp_mask: Optional[np.ndarray] = None,
    worn_tucked: bool = False,
    openpose_keypoints: Any = None,
    garment_length_cm: float = 0.0,
    user_torso_cm: float = 50.0,
) -> Image.Image:
    """
    Create a region-aware agnostic mask that expands/contracts
    different body zones independently based on fit.
    
    For oversized garments:
      - Torso region expands significantly (excess fabric at midsection)
      - Shoulder region expands moderately (dropped shoulders)
      - Arm regions extend (longer sleeves)
    
    For tight garments:
      - All regions contract toward body center
    """
    h, w = np.array(person_image).shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gap = fit_profile.size_gap
    exp = fit_profile.mask_expansion_px
    
    gtype = garment_type.lower()
    
    # [TASK 2: Original Mask Area Extraction]
    original_area = 0
    if schp_mask is not None:
        original_area = np.sum(schp_mask == 4)  # 4 is upper_clothes
    
    if densepose_map is not None:
        # Use DensePose body part IDs for precise regional control
        # IDs: 1-2=torso, 3-4=upper arms, 5-6=lower arms, 7-8=upper legs, etc.
        
        if gtype in ("top", "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket"):
            # Upper body garment
            torso = ((densepose_map == 1) | (densepose_map == 2)).astype(np.uint8) * 255
            arms = ((densepose_map == 3) | (densepose_map == 4) |
                    (densepose_map == 5) | (densepose_map == 6)).astype(np.uint8) * 255
            
            # Expand torso by full amount
            if exp > 0:
                k_torso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp * 2, exp))
                torso = cv2.dilate(torso, k_torso)
                
                k_arms = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp, int(exp * 1.5)))
                arms = cv2.dilate(arms, k_arms)
            elif exp < 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(exp), abs(exp)))
                torso = cv2.erode(torso, k)
                arms = cv2.erode(arms, k)
            
            mask = np.maximum(torso, arms)
            
        elif gtype in ("pants", "jeans", "trousers", "skirt", "tights", "leggings"):
            # Lower body garment
            legs = ((densepose_map >= 7) & (densepose_map <= 10)).astype(np.uint8) * 255
            hip = ((densepose_map == 1) | (densepose_map == 2)).astype(np.uint8) * 255
            # Only take lower part of torso for hip
            mid_y = h // 2
            hip[:mid_y, :] = 0
            
            if exp > 0:
                k_hip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp * 2, exp))
                hip = cv2.dilate(hip, k_hip)
                k_legs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp, exp))
                legs = cv2.dilate(legs, k_legs)
            elif exp < 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(exp), abs(exp)))
                hip = cv2.erode(hip, k)
                legs = cv2.erode(legs, k)
            
            mask = np.maximum(hip, legs)
            
        elif gtype in ("dress", "bodysuit"):
            # Full body
            body = (densepose_map > 0).astype(np.uint8) * 255
            if exp > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp * 2, exp))
                body = cv2.dilate(body, k)
            elif exp < 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(exp), abs(exp)))
                body = cv2.erode(body, k)
            mask = body
    else:
        # Fallback: heuristic mask based on image proportions
        if gtype in ("top", "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket"):
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)
            y1 = int(h * 0.20)  # start below chin (was 0.12 which cut into face)
            y2 = int(h * 0.65)
        elif gtype in ("pants", "jeans", "trousers", "skirt", "tights", "leggings"):
            x1 = int(w * 0.20)
            x2 = int(w * 0.80)
            y1 = int(h * 0.45)
            y2 = int(h * 0.95)
        else:  # dress
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)
            y1 = int(h * 0.12)
            y2 = int(h * 0.90)
        
        # Apply expansion
        x1 = max(0, x1 - exp)
        x2 = min(w, x2 + exp)
        y1 = max(0, y1 - (exp // 2))
        y2 = min(h, y2 + (exp // 2))
        
        mask[y1:y2, x1:x2] = 255
    
    # ── Layer 5: Apply gradient edge softness ──
    softness = fit_profile.edge_softness
    if softness > 1:
        # Make kernel size odd
        ks = softness if softness % 2 == 1 else softness + 1
        mask = cv2.GaussianBlur(mask, (ks, ks), softness / 2)
    
    # Threshold back to binary (but keep soft edges for loose fits)
    if gap >= 2:
        # For loose fits: keep gradient edges (fabric "spills" beyond body)
        pass  # Keep the blurred mask as-is (values between 0-255)
    else:
        mask = (mask > 128).astype(np.uint8) * 255
        
    # [TASK 2 MOD: Phase 2 Fix] Apply width_ratio ONLY to scale the Target Boundary Mask
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        region = mask[y1:y2, x1:x2]
        
        # Scale the mask by the standard fit structure ratios
        new_w = max(1, int((x2 - x1) * fit_profile.width_ratio))
        new_h = max(1, int((y2 - y1) * fit_profile.length_ratio))
        
        region_scaled = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center the scaled region back onto the mask canvas
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        new_mask = np.zeros_like(mask)
        
        py1 = max(0, cy - new_h // 2)
        py2 = py1 + new_h
        px1 = max(0, cx - new_w // 2)
        px2 = px1 + new_w
        
        # Safe clipping constraints
        py1_clamped = max(0, py1)
        py2_clamped = min(h, py2)
        px1_clamped = max(0, px1)
        px2_clamped = min(w, px2)
        
        src_y1 = py1_clamped - py1
        src_y2 = new_h - (py2 - py2_clamped)
        src_x1 = px1_clamped - px1
        src_x2 = new_w - (px2 - px2_clamped)
        
        if src_y2 > src_y1 and src_x2 > src_x1:
            new_mask[py1_clamped:py2_clamped, px1_clamped:px2_clamped] = region_scaled[src_y1:src_y2, src_x1:src_x2]
            
        mask = new_mask
        
    # [TASK 1 MOD: The length_ratio Vertical Displacement]
    if openpose_keypoints is not None and garment_length_cm > 0:
        points = []
        if isinstance(openpose_keypoints, dict):
            points = list(openpose_keypoints.values())
        elif isinstance(openpose_keypoints, list) or isinstance(openpose_keypoints, np.ndarray):
            points = openpose_keypoints
            
        def get_pt(idx):
            if idx < len(points):
                p = points[points_idx] if 'points_idx' in locals() else points[idx]
                if p is not None and len(p) >= 2 and p[0] > 0 and p[1] > 0:
                    return p[:2]
            return None
            
        neck = get_pt(1)
        r_hip = get_pt(8)
        l_hip = get_pt(11)
        
        hip_y = None
        if r_hip is not None and l_hip is not None:
            hip_y = (r_hip[1] + l_hip[1]) / 2.0
        elif r_hip is not None:
            hip_y = r_hip[1]
        elif l_hip is not None:
            hip_y = l_hip[1]
            
        if neck is not None and hip_y is not None:
            torso_pixel_len = max(1.0, float(hip_y - neck[1]))
            phys_length_ratio = garment_length_cm / user_torso_cm
            
            # If length dictates it should end above the hips
            if phys_length_ratio < 0.95:
                target_pixel_len = torso_pixel_len * phys_length_ratio
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_top = coords[0].min()
                    y_bottom = coords[0].max()
                    current_pixel_len = y_bottom - y_top
                    if current_pixel_len > target_pixel_len:
                        crop_pixels = int(current_pixel_len - target_pixel_len)
                        if crop_pixels > 0 and crop_pixels < (y_bottom - y_top):
                            mask[(y_bottom - crop_pixels):y_bottom, :] = 0
        
    # [TASK 2 MOD: The Underlying Bulk Guardrail]
    new_area = np.sum(mask > 0)
    if fit_profile.width_ratio < 0.90 and original_area > 0:
        if original_area > new_area * 1.30:
            fit_profile.halt_generation = True
            fit_profile.error_msg = "Source clothing too bulky for requested size reduction. Please upload a photo wearing form-fitting clothes."
            print(f"HALT: {fit_profile.error_msg}")

    # [TASK 3 MOD: Z-Index Hemline Layering]
    if schp_mask is not None:
        if worn_tucked:
            # schp label 9 = pants, 12 = skirt
            lower_body_mask = ((schp_mask == 9) | (schp_mask == 12)).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(lower_body_mask))
        else:
            # Overwrites upper boundary of pants natively (handled by IDM-VTON mask prioritization natively)
            pass

    # Hard-protect the face zone: never let the mask extend above 20% of image
    # height for upper-body garments, regardless of DensePose dilation.
    if gtype in ("top", "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket"):
        face_safe_rows = int(h * 0.20)
        mask[:face_safe_rows, :] = 0

    return Image.fromarray(mask)


# ════════════════════════════════════════════════════════════════
#  Layer 6: TPS Warp Intensity Control
# ════════════════════════════════════════════════════════════════

def apply_fit_aware_warp(
    garment_image: Image.Image,
    person_image: Image.Image,
    fit_profile: FitProfile,
    densepose_map: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Apply TPS warping with intensity controlled by the fit type.
    
    Tight fits → aggressive warping (fabric follows body contour closely)
    Loose fits → gentle warping (fabric hangs away from body)
    Oversized → minimal warping (fabric is independent of body shape)
    """
    intensity = fit_profile.warp_intensity
    
    if intensity < 0.25:
        # Almost no warping for very oversized — garment shape is independent
        return garment_image
    
    garment_np = np.array(garment_image)
    person_np = np.array(person_image)
    h, w = garment_np.shape[:2]
    
    # Define source points (garment corners and midpoints)
    src_points = np.array([
        [w * 0.3, h * 0.15],   # left shoulder
        [w * 0.7, h * 0.15],   # right shoulder
        [w * 0.2, h * 0.5],    # left waist
        [w * 0.8, h * 0.5],    # right waist
        [w * 0.25, h * 0.85],  # left hem
        [w * 0.75, h * 0.85],  # right hem
        [w * 0.5, h * 0.1],    # center neck
        [w * 0.5, h * 0.9],    # center hem
    ], dtype=np.float32)
    
    # Define target points (body contour from DensePose or heuristic)
    if densepose_map is not None:
        # Extract body contour points from DensePose
        body_mask = (densepose_map > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            # Sample points along contour at corresponding heights
            dst_points = _sample_contour_at_heights(
                contour, src_points, w, h
            )
        else:
            dst_points = src_points.copy()
    else:
        # Heuristic body contour (slightly narrower than garment, centered)
        body_w_factor = 0.85  # body is ~85% of garment width
        dst_points = np.array([
            [w * 0.32, h * 0.15],
            [w * 0.68, h * 0.15],
            [w * 0.25, h * 0.5],
            [w * 0.75, h * 0.5],
            [w * 0.28, h * 0.85],
            [w * 0.72, h * 0.85],
            [w * 0.5, h * 0.1],
            [w * 0.5, h * 0.9],
        ], dtype=np.float32)
    
    # Interpolate between source and destination based on intensity
    # intensity=1.0 → full body-hugging, intensity=0.0 → no change
    warped_points = src_points + intensity * (dst_points - src_points)
    
    # Compute transformation (TPS preferred, perspective warp as fallback)
    src_reshaped = src_points.reshape(1, -1, 2)
    dst_reshaped = warped_points.reshape(1, -1, 2)
    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
        tps.estimateTransformation(dst_reshaped, src_reshaped, matches)
        result = tps.warpImage(garment_np)
    except AttributeError:
        # opencv-contrib not available; fall back to perspective warp
        src_corners = np.array(
            [src_points[0], src_points[1], src_points[4], src_points[5]],
            dtype=np.float32,
        )
        dst_corners = np.array(
            [warped_points[0], warped_points[1], warped_points[4], warped_points[5]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        result = cv2.warpPerspective(garment_np, M, (w, h))
    
    return Image.fromarray(result)


def _sample_contour_at_heights(contour, src_points, w, h):
    """Sample contour points at the same vertical positions as src_points."""
    dst = src_points.copy()
    contour_pts = contour.reshape(-1, 2).astype(float)
    
    for i, (sx, sy) in enumerate(src_points):
        # Find contour points near this height (y ± tolerance)
        tolerance = h * 0.05
        nearby = contour_pts[np.abs(contour_pts[:, 1] - sy) < tolerance]
        
        if len(nearby) > 0:
            if sx < w / 2:
                # Left side: find leftmost contour point
                dst[i, 0] = nearby[:, 0].min()
            else:
                # Right side: find rightmost contour point
                dst[i, 0] = nearby[:, 0].max()
            dst[i, 1] = sy  # Keep same height
    
    return dst


# ════════════════════════════════════════════════════════════════
#  Layer 7: The Complete Engine
# ════════════════════════════════════════════════════════════════

class SizeAwareVTON:
    """
    The complete size-aware VTON engine.
    
    Orchestrates all 7 physics layers to transform a standard
    IDM-VTON model into a size-aware system.
    """
    
    def __init__(self, target_resolution: Tuple[int, int] = (768, 1024)):
        self.target_w, self.target_h = target_resolution
    
    def prepare_size_aware_inputs(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        person_size: str = "M",
        garment_size: str = "M",
        garment_type: str = "top",
        densepose_map: Optional[np.ndarray] = None,
        schp_mask: Optional[np.ndarray] = None,
        worn_tucked: bool = False,
        openpose_keypoints: Any = None,
        garment_length_cm: float = 0.0,
        user_torso_cm: float = 50.0,
        neckline_type: str = "",
        fabric_rigidity: float = 0.5,
    ) -> Dict:
        """
        Prepare all size-aware inputs for IDM-VTON inference.
        
        Returns a dict with everything needed:
            - resized_garment: physically resized garment image
            - warped_garment: TPS-warped garment (fit-aware intensity)
            - agnostic_mask: region-aware, gradient-edged mask
            - positive_prompt: physics-descriptive text
            - negative_prompt: anti-descriptors
            - inpainting_strength: model creative freedom
            - fit_profile: full fit analysis
        """
        # Step 1: Classify the fit
        fit = classify_fit(garment_type, garment_size, person_size, fabric_rigidity)

        # Step 2: Resize garment mathematically
        resized = resize_garment(garment_image, fit, garment_type)
        
        # Step 3: Apply fit-aware TPS warping
        warped = apply_fit_aware_warp(resized, person_image, fit, densepose_map)
        
        # Step 4: Create regional mask with gradient edges
        mask = create_regional_mask(person_image, densepose_map, fit, garment_type, schp_mask, worn_tucked, openpose_keypoints, garment_length_cm, user_torso_cm)

        # Step 4b: DensePose torso-floor pelvis clamp — prevents tee-to-dress hallucination.
        # Finds the lowest Y pixel of the DensePose torso region (labels 1 & 2) and zeros
        # out every mask row below that point + 5% padding.  Runs only for top garments.
        # Fails silently: any exception leaves the mask completely unmodified.
        if garment_type.lower() in ("top", "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket"):
            try:
                # Normalise densepose_map to a 2-D uint8 numpy array
                if densepose_map is not None:
                    if isinstance(densepose_map, Image.Image):
                        dp_arr = np.array(densepose_map.convert("L"))
                    else:
                        dp_arr = np.array(densepose_map)
                    # Collapse to 2-D if the array has a channel dimension
                    if dp_arr.ndim == 3:
                        # DensePose part-label maps are usually single-channel stored as RGB;
                        # take the first channel which holds the part index.
                        dp_arr = dp_arr[:, :, 0]

                    # DensePose COCO part IDs for torso: 1 (torso-front) and 2 (torso-back).
                    # Any non-zero overlap means a body pixel; we use all non-zero as a broad
                    # torso proxy when the exact label range is uncertain (segm vs IUV maps).
                    torso_mask = (dp_arr == 1) | (dp_arr == 2)
                    if not torso_mask.any():
                        # Broad fallback: treat every non-zero DensePose pixel as body
                        torso_mask = dp_arr > 0

                    if torso_mask.any():
                        # Find the lowest (max Y) torso pixel row, scale to mask dimensions
                        dp_h, dp_w = dp_arr.shape[:2]
                        torso_max_y_dp = int(np.max(np.where(torso_mask)[0]))  # row index in dp space

                        # Convert to mask pixel space (mask is still at person_image resolution here)
                        mask_arr = np.array(mask)
                        mask_h = mask_arr.shape[0]
                        torso_max_y_mask = int(torso_max_y_dp * mask_h / dp_h)

                        # 5% downward padding so the hemline sits naturally below the literal waist
                        padding_px = int(mask_h * 0.05)
                        cut_y = min(mask_h, torso_max_y_mask + padding_px)

                        mask_arr[cut_y:, :] = 0
                        mask = Image.fromarray(mask_arr)
            except Exception:
                pass  # leave mask unmodified on any failure

        # Step 5: Resize all to target resolution
        warped = warped.resize((self.target_w, self.target_h), Image.LANCZOS)
        mask = mask.resize((self.target_w, self.target_h), Image.NEAREST)
        person_resized = person_image.resize((self.target_w, self.target_h), Image.LANCZOS)
        
        return {
            "person_image": person_resized,
            "garment_image": warped,
            "garment_image_no_warp": resized.resize((self.target_w, self.target_h), Image.LANCZOS),
            "agnostic_mask": mask,
            "positive_prompt": fit.positive_prompt,
            "negative_prompt": fit.negative_prompt,
            "inpainting_strength": fit.inpainting_strength,
            "fit_profile": fit,
        }
        
    def apply_post_processing(
        self,
        original_person_image: Image.Image,
        generated_image: Image.Image,
        schp_mask: np.ndarray,
    ) -> Image.Image:
        """Pass-through — all compositing removed; return raw generated image."""
        w, h = self.target_w, self.target_h
        return generated_image.resize((w, h), Image.LANCZOS)
    
    def generate_comparison(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        person_size: str = "M",
        garment_type: str = "top",
        sizes: List[str] = None,
    ) -> Image.Image:
        """
        Generate a visual comparison showing how the garment preprocessing
        changes across different sizes.
        
        Shows: resized garment + mask + fit description for each size.
        """
        if sizes is None:
            sizes = ["S", "M", "L", "XL", "2XL"]
        
        panels = []
        for gs in sizes:
            result = self.prepare_size_aware_inputs(
                person_image=person_image.copy(),
                garment_image=garment_image.copy(),
                person_size=person_size,
                garment_size=gs,
                garment_type=garment_type,
            )
            
            fit = result["fit_profile"]
            
            # Create panel with garment + info overlay
            garment_np = np.array(result["garment_image"])
            
            # Add text overlays
            y = 25
            texts = [
                f"Garment: {gs}",
                f"Fit: {fit.fit_type.name}",
                f"W: {fit.width_ratio:.2f}x",
                f"Warp: {fit.warp_intensity:.1f}",
                f"Tension: {fit.fabric_tension:.2f}",
                f"Gap: {fit.size_gap:+d}",
            ]
            for txt in texts:
                cv2.putText(garment_np, txt, (8, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(garment_np, txt, (8, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 22
            
            panels.append(garment_np)
        
        comparison = np.concatenate(panels, axis=1)
        return Image.fromarray(comparison)


# ════════════════════════════════════════════════════════════════
#  Self-test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  FitFusion — Size-Aware VTON Engine Self-Test")
    print("=" * 60)
    
    engine = SizeAwareVTON()
    
    # Test fit classification for all size gaps
    print("\nFit profiles (Size M person):\n")
    print(f"{'Grmt':>5} {'Fit Type':<12} {'W-Ratio':>8} {'Warp':>6} "
          f"{'Mask-px':>8} {'Strength':>9} {'Edge':>5} {'Tension':>8}")
    print("-" * 75)
    
    for gs in ["XS", "S", "M", "L", "XL", "2XL", "3XL"]:
        fit = classify_fit("top", gs, "M")
        print(f"{gs:>5} {fit.fit_type.name:<12} {fit.width_ratio:>8.3f} "
              f"{fit.warp_intensity:>6.2f} {fit.mask_expansion_px:>+8d} "
              f"{fit.inpainting_strength:>9.2f} {fit.edge_softness:>5d} "
              f"{fit.fabric_tension:>8.2f}")
    
    print("\n\nPrompt examples:")
    for gs in ["S", "M", "XL"]:
        fit = classify_fit("top", gs, "M")
        print(f"\n  {gs} on M person ({fit.fit_type.name}):")
        print(f"    + {fit.positive_prompt[:80]}...")
        print(f"    - {fit.negative_prompt[:80]}...")
    
    # Test with dummy images
    print("\n\nGenerating visual comparison...")
    dummy_garment = Image.new("RGB", (768, 1024), (255, 255, 255))
    gn = np.array(dummy_garment)
    cv2.rectangle(gn, (250, 200), (518, 700), (50, 100, 200), -1)
    cv2.rectangle(gn, (150, 200), (250, 400), (50, 100, 200), -1)
    cv2.rectangle(gn, (518, 200), (618, 400), (50, 100, 200), -1)
    dummy_garment = Image.fromarray(gn)
    
    dummy_person = Image.new("RGB", (768, 1024), (200, 200, 200))
    
    comparison = engine.generate_comparison(
        dummy_person, dummy_garment, "M", "top",
        ["XS", "S", "M", "L", "XL", "2XL"]
    )
    comparison.save("size_aware_comparison.png")
    print("Saved to size_aware_comparison.png")
    print("\n✓ Self-test complete!")
