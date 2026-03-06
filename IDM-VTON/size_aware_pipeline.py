"""
FitFusion — Size-Aware VTON Pipeline (Physics-Based)
=====================================================
Uses physics and mathematics to pre-process garment inputs
so that the pre-trained IDM-VTON model produces size-correct
try-on images WITHOUT any additional training.

The key insight: instead of teaching the AI what "size" means,
we physically resize the garment image and mask to reflect the
correct size relationship, then let the model composite naturally.

Pipeline:
    1. Compute size ratio (garment size ÷ person size)
    2. Resize flat garment image by that ratio
    3. TPS-warp toward person's body contour
    4. Generate size-adaptive agnostic mask
    5. Run IDM-VTON inference with modified inputs
    6. Post-process output

Usage:
    from size_aware_pipeline import SizeAwarePipeline
    
    pipeline = SizeAwarePipeline()
    result = pipeline.run(
        person_image=person_img,       # PIL Image
        garment_image=garment_img,     # PIL Image (flat product photo)
        person_size="M",               # Person's actual size
        garment_size="XL",             # Garment's labeled size
        garment_type="top",            # top, pants, dress, etc.
    )
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Dict

from size_charts import compute_size_ratio, get_garment_dimensions


class GarmentResizer:
    """
    Physically resizes a flat garment image based on the mathematical
    ratio between garment size and person size.
    
    This is the core of the physics-based approach: instead of hoping
    the AI understands "XL," we make the garment image actually wider.
    """
    
    def __init__(self, target_resolution: Tuple[int, int] = (768, 1024)):
        """
        Args:
            target_resolution: (width, height) of the final image for IDM-VTON
        """
        self.target_w, self.target_h = target_resolution
    
    def resize_garment(
        self,
        garment_image: Image.Image,
        garment_type: str,
        garment_size: str,
        person_size: str,
        fit_preference: str = "regular",  # "tight", "regular", "loose"
    ) -> Image.Image:
        """
        Resize the garment image based on the size ratio.
        
        Args:
            garment_image: flat product photo of the garment
            garment_type: "top", "pants", "dress", etc.
            garment_size: the garment's labeled size ("S", "M", "XL")
            person_size: the person's actual body size
            fit_preference: how the user wants the garment to fit
        
        Returns:
            Resized garment image (same canvas size, garment scaled within)
        """
        # 1. Compute the mathematical size ratio
        ratio = compute_size_ratio(garment_type, garment_size, person_size)
        width_ratio = ratio["width_ratio"]
        length_ratio = ratio["length_ratio"]
        
        # 2. Apply fit preference modifier
        fit_modifiers = {
            "tight": 0.95,      # 5% tighter than standard
            "regular": 1.0,     # Standard fit
            "loose": 1.08,      # 8% looser than standard
            "oversized": 1.15,  # 15% oversized
        }
        fit_mod = fit_modifiers.get(fit_preference, 1.0)
        width_ratio *= fit_mod
        length_ratio *= fit_mod
        
        # 3. Convert garment to numpy for processing
        garment_np = np.array(garment_image)
        orig_h, orig_w = garment_np.shape[:2]
        
        # 4. Find the garment bounding box (non-white/non-transparent pixels)
        garment_mask = self._extract_garment_mask(garment_np)
        bbox = self._get_bounding_box(garment_mask)
        
        if bbox is None:
            # Can't find garment, return as-is
            return garment_image
        
        x1, y1, x2, y2 = bbox
        garment_region = garment_np[y1:y2, x1:x2]
        garment_region_mask = garment_mask[y1:y2, x1:x2]
        
        region_h, region_w = garment_region.shape[:2]
        
        # 5. Compute new dimensions based on ratio
        new_w = int(region_w * width_ratio)
        new_h = int(region_h * length_ratio)
        
        # Clamp to reasonable bounds (don't exceed canvas)
        new_w = min(new_w, orig_w - 20)
        new_h = min(new_h, orig_h - 20)
        new_w = max(new_w, int(region_w * 0.5))  # Don't shrink below 50%
        new_h = max(new_h, int(region_h * 0.5))
        
        # 6. Resize the garment region
        resized_region = cv2.resize(
            garment_region, (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        resized_mask = cv2.resize(
            garment_region_mask, (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # 7. Create output canvas (white background)
        output = np.ones_like(garment_np) * 255
        
        # Center the resized garment on the canvas
        cx = orig_w // 2
        cy = orig_h // 2
        
        paste_x1 = max(0, cx - new_w // 2)
        paste_y1 = max(0, cy - new_h // 2)
        paste_x2 = min(orig_w, paste_x1 + new_w)
        paste_y2 = min(orig_h, paste_y1 + new_h)
        
        # Adjust region if it was clipped
        src_x2 = paste_x2 - paste_x1
        src_y2 = paste_y2 - paste_y1
        
        # Composite: place garment pixels where mask is active
        region_crop = resized_region[:src_y2, :src_x2]
        mask_crop = resized_mask[:src_y2, :src_x2]
        
        canvas_region = output[paste_y1:paste_y2, paste_x1:paste_x2]
        mask_3ch = np.stack([mask_crop] * 3, axis=2) if len(garment_np.shape) == 3 else mask_crop
        
        canvas_region[mask_3ch > 128] = region_crop[mask_3ch > 128]
        output[paste_y1:paste_y2, paste_x1:paste_x2] = canvas_region
        
        return Image.fromarray(output)
    
    def _extract_garment_mask(self, img: np.ndarray, threshold: int = 240) -> np.ndarray:
        """
        Extract a binary mask of the garment (non-white, non-transparent pixels).
        """
        if img.shape[2] == 4:
            # RGBA — use alpha channel
            return (img[:, :, 3] > 128).astype(np.uint8) * 255
        
        # RGB — threshold against white background
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = (gray < threshold).astype(np.uint8) * 255
        
        # Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _get_bounding_box(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get the bounding box of non-zero pixels in the mask."""
        coords = np.where(mask > 128)
        if len(coords[0]) == 0:
            return None
        
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        # Add small padding
        pad = 5
        y1 = max(0, y1 - pad)
        x1 = max(0, x1 - pad)
        y2 = min(mask.shape[0], y2 + pad)
        x2 = min(mask.shape[1], x2 + pad)
        
        return (x1, y1, x2, y2)


class MaskAdapter:
    """
    Adapts the agnostic mask based on the size ratio.
    
    When a garment is LARGER than the person → expand mask area
    (model needs more canvas space to render excess fabric)
    
    When a garment is SMALLER → slightly contract mask
    (model should render tighter-fitting garment)
    """
    
    def adapt_mask(
        self,
        agnostic_mask: Image.Image,
        garment_type: str,
        garment_size: str,
        person_size: str,
    ) -> Image.Image:
        """
        Adapt the agnostic mask based on the size relationship.
        
        Args:
            agnostic_mask: original body-region mask
            garment_type, garment_size, person_size: for ratio computation
        
        Returns:
            Adapted mask image
        """
        ratio = compute_size_ratio(garment_type, garment_size, person_size)
        size_gap = ratio["size_gap"]
        width_ratio = ratio["width_ratio"]
        
        mask_np = np.array(agnostic_mask.convert("L"))
        
        if abs(size_gap) == 0:
            return agnostic_mask
        
        if size_gap > 0:
            # Garment is LARGER → dilate mask
            # Scale dilation by the magnitude of the size gap
            kernel_size = min(int(abs(width_ratio - 1.0) * 100), 40)
            kernel_size = max(kernel_size, 3)
            
            if garment_type.lower() in ("top", "shirt", "hoodie", "jacket", "blouse", "sweater", "t-shirt", "tee"):
                # Upper body: dilate more horizontally
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size * 2, kernel_size)
                )
            elif garment_type.lower() in ("pants", "jeans", "trousers", "tights", "leggings"):
                # Lower body: dilate more vertically
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size * 2)
                )
            else:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
            
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            
        else:
            # Garment is SMALLER → erode mask slightly
            kernel_size = min(int(abs(width_ratio - 1.0) * 50), 15)
            kernel_size = max(kernel_size, 3)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            mask_np = cv2.erode(mask_np, kernel, iterations=1)
        
        # Smooth edges
        mask_np = cv2.GaussianBlur(mask_np, (5, 5), 2)
        mask_np = (mask_np > 128).astype(np.uint8) * 255
        
        return Image.fromarray(mask_np)


class SizeAwarePipeline:
    """
    End-to-end physics-based size-aware VTON pipeline.
    
    Orchestrates:
        1. Size ratio calculation
        2. Garment image resizing
        3. Mask adaptation
        4. IDM-VTON inference (when connected to model)
    """
    
    def __init__(self, target_resolution: Tuple[int, int] = (768, 1024)):
        self.resizer = GarmentResizer(target_resolution)
        self.mask_adapter = MaskAdapter()
        self.target_w, self.target_h = target_resolution
    
    def prepare_inputs(
        self,
        garment_image: Image.Image,
        agnostic_mask: Image.Image,
        garment_type: str = "top",
        garment_size: str = "M",
        person_size: str = "M",
        fit_preference: str = "regular",
    ) -> Dict:
        """
        Prepare size-adjusted inputs for IDM-VTON inference.
        
        This is the core function — it takes the original garment image
        and mask, and returns modified versions that encode the size
        relationship physically.
        
        Args:
            garment_image: flat product photo
            agnostic_mask: body-region mask from the person image
            garment_type: "top", "pants", "dress", etc.
            garment_size: the garment's labeled size
            person_size: the person's body size
            fit_preference: "tight", "regular", "loose", "oversized"
        
        Returns:
            dict with:
                "garment_image": resized garment PIL Image
                "agnostic_mask": adapted mask PIL Image
                "size_info": dict with ratio details
        """
        # 1. Compute the mathematical ratio
        size_info = compute_size_ratio(garment_type, garment_size, person_size)
        
        # 2. Physically resize the garment
        resized_garment = self.resizer.resize_garment(
            garment_image=garment_image,
            garment_type=garment_type,
            garment_size=garment_size,
            person_size=person_size,
            fit_preference=fit_preference,
        )
        
        # 3. Adapt the agnostic mask
        adapted_mask = self.mask_adapter.adapt_mask(
            agnostic_mask=agnostic_mask,
            garment_type=garment_type,
            garment_size=garment_size,
            person_size=person_size,
        )
        
        # 4. Resize both to target resolution
        resized_garment = resized_garment.resize(
            (self.target_w, self.target_h), Image.LANCZOS
        )
        adapted_mask = adapted_mask.resize(
            (self.target_w, self.target_h), Image.NEAREST
        )
        
        return {
            "garment_image": resized_garment,
            "agnostic_mask": adapted_mask,
            "size_info": size_info,
        }
    
    def visualize_size_comparison(
        self,
        garment_image: Image.Image,
        agnostic_mask: Image.Image,
        garment_type: str,
        person_size: str,
        sizes_to_compare: list = None,
    ) -> Image.Image:
        """
        Generate a side-by-side comparison showing how the garment
        image changes across different sizes.
        
        This is useful for debugging and demonstrating size-awareness.
        
        Args:
            garment_image: original flat garment
            agnostic_mask: original mask
            garment_type: type of garment
            person_size: the person's size
            sizes_to_compare: list of sizes to show (default: S through 2XL)
        
        Returns:
            PIL Image with side-by-side comparison
        """
        if sizes_to_compare is None:
            sizes_to_compare = ["S", "M", "L", "XL", "2XL"]
        
        panels = []
        for gs in sizes_to_compare:
            result = self.prepare_inputs(
                garment_image=garment_image.copy(),
                agnostic_mask=agnostic_mask.copy(),
                garment_type=garment_type,
                garment_size=gs,
                person_size=person_size,
            )
            
            # Add size label to image
            panel = np.array(result["garment_image"])
            cv2.putText(
                panel, f"Size: {gs}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2
            )
            
            ratio = result["size_info"]
            cv2.putText(
                panel, f"W: {ratio['width_ratio']:.2f}x",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 200), 2
            )
            
            panels.append(panel)
        
        # Stack horizontally
        comparison = np.concatenate(panels, axis=1)
        return Image.fromarray(comparison)


if __name__ == "__main__":
    # Quick self-test with a dummy garment
    print("=== Size-Aware Pipeline Self-Test ===\n")
    
    # Create a simple test garment (white background with colored rectangle)
    garment = Image.new("RGB", (768, 1024), (255, 255, 255))
    garment_np = np.array(garment)
    # Draw a T-shirt shape
    cv2.rectangle(garment_np, (250, 200), (518, 700), (50, 100, 200), -1)  # body
    cv2.rectangle(garment_np, (150, 200), (250, 400), (50, 100, 200), -1)  # left sleeve
    cv2.rectangle(garment_np, (518, 200), (618, 400), (50, 100, 200), -1)  # right sleeve
    garment = Image.fromarray(garment_np)
    
    # Create a dummy mask
    mask = Image.new("L", (768, 1024), 0)
    mask_np = np.array(mask)
    cv2.rectangle(mask_np, (200, 150), (568, 750), 255, -1)
    mask = Image.fromarray(mask_np)
    
    pipeline = SizeAwarePipeline()
    
    print("Person size: M\n")
    for gs in ["S", "M", "L", "XL", "2XL"]:
        result = pipeline.prepare_inputs(
            garment_image=garment.copy(),
            agnostic_mask=mask.copy(),
            garment_type="top",
            garment_size=gs,
            person_size="M",
        )
        info = result["size_info"]
        print(f"  Garment size {gs}:")
        print(f"    Width ratio:  {info['width_ratio']:.3f}x ({info['garment_width_cm']}cm / {info['person_width_cm']}cm)")
        print(f"    Length ratio: {info['length_ratio']:.3f}x ({info['garment_length_cm']}cm / {info['person_length_cm']}cm)")
        print(f"    Size gap:     {info['size_gap']:+d} sizes")
        print()
    
    # Generate comparison image
    comparison = pipeline.visualize_size_comparison(
        garment, mask, "top", "M", ["S", "M", "L", "XL", "2XL"]
    )
    comparison.save("size_comparison_test.png")
    print("Saved comparison to size_comparison_test.png")
