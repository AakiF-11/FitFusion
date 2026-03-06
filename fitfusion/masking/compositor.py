import numpy as np
from PIL import Image
import cv2

def restore_original_skin(original_image: Image.Image, generated_image: Image.Image, schp_mask: np.ndarray) -> Image.Image:
    """
    Layers the original skin back over the generated image using the SCHP segmentation labels.
    Prevents the diffusion model from hallucinating generic skin tones or erasing the user's tattoos.
    Skin labels used: 'face', 'neck', 'left_arm', 'right_arm', 'left_leg', 'right_leg'.
    """
    orig_np = np.array(original_image.convert("RGB"))
    gen_np = np.array(generated_image.convert("RGB"))
    
    if orig_np.shape != gen_np.shape:
        gen_np = cv2.resize(gen_np, (orig_np.shape[1], orig_np.shape[0]))
        
    if schp_mask.shape[:2] != orig_np.shape[:2]:
        schp_mask = cv2.resize(schp_mask, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Standard SCHP Lip parsing indices:
    # Face=13, Left Arm=14, Right Arm=15, Left Leg=16, Right Leg=17, Neck=10
    # Left Hand and Right Hand are captured within arms (14, 15) or explicitly as gloves(3)
    skin_labels = [10, 13, 14, 15, 16, 17]
    hand_labels = [3, 14, 15] # 3=Glove, 14=Left Arm/Hand, 15=Right Arm/Hand
    
    # Create binary mask for general skin regions
    skin_mask = np.isin(schp_mask, skin_labels).astype(np.uint8) * 255
    # Slight blur to mask for seamless blending
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin_mask_3d = np.stack([skin_mask / 255.0] * 3, axis=2)
    
    # [TASK 1: Absolute Hand Exclusion Zone]
    # Explicitly extract the left_hand and right_hand indices from the SCHP mask
    hand_mask = np.isin(schp_mask, hand_labels).astype(np.uint8) * 255
    hand_mask = cv2.GaussianBlur(hand_mask, (1, 1), 0) # Less blur, unconditional hard mask
    hand_mask_3d = np.stack([hand_mask / 255.0] * 3, axis=2)
    
    # Paste the original_image's skin pixels over the generated_image using alpha compositing
    final_np = (orig_np * skin_mask_3d + gen_np * (1.0 - skin_mask_3d)).astype(np.float32)
    
    # Final compositing step: unconditionally paste raw hand pixels to crush hallucinated fingers
    final_np = (orig_np * hand_mask_3d + final_np * (1.0 - hand_mask_3d)).astype(np.uint8)
    
    return Image.fromarray(final_np)
