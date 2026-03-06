import os
import sys
import numpy as np
import cv2
from PIL import Image

# Route imports correctly for the workspace
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "IDM-VTON"))

try:
    import size_charts
    from size_aware_vton import classify_fit, create_regional_mask
    from run_tryon import _do_inference
except ImportError as e:
    print(f"Critical module import failed. Please ensure environment paths are correct: {e}")
    sys.exit(1)

def build_evaluation_matrix():
    print("=" * 60)
    print("  FitFusion — Phase 2 Physics Engine Matrix Test")
    print("=" * 60)

    # Validate or instantiate the baseline images
    customer_photo_path = os.path.join(parent_dir, "customer_photo.jpg")
    garment_photo_path = os.path.join(parent_dir, "garment.jpg")
    output_dir = os.path.join(parent_dir, "tests", "matrix_output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(customer_photo_path):
        Image.new("RGB", (768, 1024), color=(180, 180, 180)).save(customer_photo_path)
    if not os.path.exists(garment_photo_path):
        Image.new("RGB", (768, 1024), color=(255, 200, 200)).save(garment_photo_path)

    # Task 1: The Matrix Setup (Variables)
    customer_measurements = {"Waist": 80.0}
    
    garment_variants = {
        "XS": {"Waist": 65, "gap": -2},
        "S":  {"Waist": 75, "gap": -1},
        "M":  {"Waist": 80, "gap": 0},
        "L":  {"Waist": 95, "gap": 1},
        "XL": {"Waist": 110, "gap": 2},
    }
    
    print(f"Control Customer Waist: {customer_measurements['Waist']}cm\n")

    # Intercept compute_size_ratio to inject our rigid mathematical bounds without editing core
    original_compute_size_ratio = size_charts.compute_size_ratio
    def mocked_compute_size_ratio(garment_type, garment_size, person_size):
        variant = garment_variants[garment_size]
        width_ratio = variant["Waist"] / customer_measurements["Waist"]
        return {
            "size_gap": variant["gap"],
            "width_ratio": width_ratio,
            "length_ratio": 1.0
        }
    size_charts.compute_size_ratio = mocked_compute_size_ratio

    person_image = Image.open(customer_photo_path).convert("RGB")
    person_image = person_image.resize((768, 1024))
    
    garment_image = Image.open(garment_photo_path).convert("RGB")
    garment_image = garment_image.resize((768, 1024))
    
    generated_panels = []

    # Task 2: The Execution Loop
    for size, stats in garment_variants.items():
        print(f"▶ Step: {size} (Mock Waist: {stats['Waist']}cm)")
        
        # 1. Phase 2 width_ratio scaling logic
        fit_profile = classify_fit("top", size, "M")
        
        # 2. Extract Scaled Agnostic Mask 
        mask_pil = create_regional_mask(
            person_image=person_image,
            densepose_map=None,
            fit_profile=fit_profile,
            garment_type="top",
            schp_mask=None,
            worn_tucked=False
        )
        mask_pil = mask_pil.resize((768, 1024), Image.NEAREST)
        
        # Save raw OpenCV intermediate masks for explicit geometric debugging
        mask_save_path = os.path.join(output_dir, f"raw_mask_{size}.png")
        mask_pil.save(mask_save_path)
        
        # Standardize untouched pristine garment image
        pristine_garment_path = os.path.join(output_dir, f"pristine_garment_{size}.jpg")
        garment_image.save(pristine_garment_path)
        
        output_image_path = os.path.join(output_dir, f"tryon_{size}.jpg")
        
        # 3. IDM-VTON Generator Subroutine
        try:
            print(f"    Injecting to _do_inference... (Ratio: {fit_profile.width_ratio:.2f})")
            _do_inference(
                person_image_path=customer_photo_path,
                garment_image_path=pristine_garment_path,
                agnostic_mask_path=mask_save_path,
                positive_prompt=fit_profile.positive_prompt,
                negative_prompt=fit_profile.negative_prompt,
                output_path=output_image_path,
                inpainting_strength=fit_profile.inpainting_strength,
                num_steps=20, 
            )
            final_img = cv2.imread(output_image_path)
        except Exception as e:
            print(f"    [!] IDM-VTON Inference Fallback: GPU execution failed or intercepted. Constructing logical blank array. {e}")
            final_img = np.zeros((1024, 768, 3), dtype=np.uint8)
            cv2.rectangle(final_img, (0, 0), (768, 1024), (220, 220, 220), -1)
            # Imprint mask boundary on blank for verification
            mask_np = np.array(mask_pil)
            final_img[mask_np > 128] = (100, 200, 100)
            
        # Draw dynamic header tag calculations onto image payload
        label_size = f"Size: {size}"
        label_ratio = f"Ratio: {fit_profile.width_ratio:.2f}"
        
        # Drop shadow styling for text depth
        cv2.putText(final_img, label_size, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(final_img, label_size, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(final_img, label_ratio, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(final_img, label_ratio, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 200), 2)
        
        cv2.imwrite(output_image_path, final_img)
        generated_panels.append(final_img)

    # Task 3: Visual Concatenation
    if generated_panels:
        print("\nStitching final mathematical validation grid...")
        validation_grid = np.concatenate(generated_panels, axis=1)
        grid_output = os.path.join(output_dir, "validation_grid.jpg")
        cv2.imwrite(grid_output, validation_grid)
        print(f"SUCCESS: Pipeline matrix successfully rendered to: {grid_output}")

    # Revert monkeypatch closure
    size_charts.compute_size_ratio = original_compute_size_ratio

if __name__ == "__main__":
    build_evaluation_matrix()
