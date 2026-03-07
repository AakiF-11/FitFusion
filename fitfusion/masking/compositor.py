import numpy as np
from PIL import Image
import cv2

# ATR dataset label mapping (IDM-VTON humanparsing output):
#   0=Background, 1=Hat, 2=Hair, 3=Sunglasses,
#   4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress,
#   8=Belt, 9=Left shoe, 10=Right shoe, 11=Face,
#   12=Left leg, 13=Right leg, 14=Left arm, 15=Right arm,
#   16=Bag, 17=Scarf, 18=Neck (added from LIP)
_SKIN_LABELS = [11, 14, 15, 12, 13, 18]  # Face, Arms, Legs, Neck
_HAND_LABELS = [14, 15]                   # Left arm, Right arm (include hands)

def restore_original_skin(original_image: Image.Image, generated_image: Image.Image, schp_mask: np.ndarray) -> Image.Image:
    """
    Layers the original skin back over the generated image using ATR segmentation labels.
    Prevents the diffusion model from hallucinating skin tones or erasing tattoos.
    Restores: Face (11), Left Arm (14), Right Arm (15), Left Leg (12), Right Leg (13), Neck (18).
    """
    orig_np = np.array(original_image.convert("RGB"))
    gen_np  = np.array(generated_image.convert("RGB"))

    if orig_np.shape != gen_np.shape:
        gen_np = cv2.resize(gen_np, (orig_np.shape[1], orig_np.shape[0]))

    if schp_mask.shape[:2] != orig_np.shape[:2]:
        schp_mask = cv2.resize(schp_mask, (orig_np.shape[1], orig_np.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # ── Skin region mask ─────────────────────────────────────────────────────
    skin_mask = np.isin(schp_mask, _SKIN_LABELS).astype(np.uint8) * 255
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin_mask_3d = np.stack([skin_mask / 255.0] * 3, axis=2)

    # ── Arm/hand mask (hard — prevents hallucinated fingers) ─────────────────
    hand_mask = np.isin(schp_mask, _HAND_LABELS).astype(np.uint8) * 255
    hand_mask = cv2.GaussianBlur(hand_mask, (1, 1), 0)
    hand_mask_3d = np.stack([hand_mask / 255.0] * 3, axis=2)

    # Alpha-composite: skin regions use original pixels
    final_np = (orig_np * skin_mask_3d + gen_np * (1.0 - skin_mask_3d)).astype(np.float32)

    # Hard-paste arm/hand pixels to stop hallucinated fingers
    final_np = (orig_np * hand_mask_3d + final_np * (1.0 - hand_mask_3d)).astype(np.uint8)

    return Image.fromarray(final_np)

