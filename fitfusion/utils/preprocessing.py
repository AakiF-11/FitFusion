import os
from rembg import remove
from PIL import Image
from typing import Any, Tuple
import numpy as np
import cv2

def standardize_background(input_image_path: str) -> str:
    """
    Strips the background and replaces the transparent alpha channel with a solid studio gray color (R:238, G:238, B:238).
    Saves and returns the path to the new, clean image.
    """
    input_img = Image.open(input_image_path).convert("RGBA")
    
    # Remove background
    no_bg_img = remove(input_img)
    
    # Create solid gray background
    gray_bg = Image.new("RGBA", no_bg_img.size, (238, 238, 238, 255))
    
    # Composite the foreground over the gray studio background
    clean_img = Image.alpha_composite(gray_bg, no_bg_img).convert("RGB")
    
    # Save to a new path
    base, ext = os.path.splitext(input_image_path)
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
