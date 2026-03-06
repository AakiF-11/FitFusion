"""
FitFusion — Try-On API
========================
API endpoint that handles the "Try On" button on any product page.

The brand integrates a single button on their website:
    <button onclick="FitFusion.tryOn('borg-aviator-black')">✨ Try On</button>

When clicked, the system:
1. Auto-detects the garment (type, sizes, reference photos)
2. Asks the customer for their photo + measurements
3. Shows the garment on their body in the selected size
4. Customer toggles sizes → instant visual comparison on THEIR body

API Flow:
    POST /tryon/init      → product_id → returns garment info + available sizes
    POST /tryon/generate   → product_id + customer_photo + measurements + size
                           → returns try-on image

Usage:
    api = TryOnAPI(catalog_dir="data/brand_catalog")
    
    # Step 1: Customer clicks "Try On" on product page
    info = api.init_tryon(product_id="borg-aviator-black", brand_id="snag_tights")
    
    # Step 2: Customer submits photo + measurements + size
    result = api.generate_tryon(
        product_id="borg-aviator-black",
        brand_id="snag_tights",
        customer_photo="customer.jpg",
        bust_cm=92, waist_cm=76, hips_cm=100, height_cm=170,
        selected_size="XL",
    )
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, List
from PIL import Image

from brand_catalog import BrandCatalog
from size_aware_vton import SizeAwareVTON, classify_fit
from size_charts import compute_size_ratio


class TryOnAPI:
    """
    API layer for the "Try On" button integration.
    
    Brands embed a button → customer clicks → we handle everything.
    """
    
    def __init__(
        self,
        catalog_dir: str = "data/brand_catalog",
        output_dir: str = "data/tryon_output",
    ):
        self.catalog = BrandCatalog(catalog_dir)
        self.physics = SizeAwareVTON()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ──────────────────────────────────────────────────────────
    #  Step 1: Init — Customer clicks "Try On" button
    # ──────────────────────────────────────────────────────────
    
    def init_tryon(self, product_id: str, brand_id: str) -> Dict:
        """
        Called when customer clicks "Try On" on a product page.
        
        The brand's website sends: product_id + brand_id
        We return: garment info, available sizes, and what we need from them.
        
        Args:
            product_id: the product's unique ID from the brand's catalog
            brand_id: the brand identifier
        
        Returns:
            {
                "status": "ready",
                "garment": {
                    "name": "Borg Aviator Jacket",
                    "type": "jacket",
                    "available_sizes": ["S", "M", "L", "XL", "2XL"],
                    "thumbnail": "path/to/image.jpg"
                },
                "requires": {
                    "customer_photo": true,
                    "measurements": ["bust_cm", "waist_cm", "hips_cm", "height_cm"]
                }
            }
        """
        full_id = f"{brand_id}_{product_id}"
        garment = self.catalog.garments.get(full_id)
        
        if not garment:
            return {
                "status": "error",
                "message": f"Product '{product_id}' not found in brand '{brand_id}'. "
                           f"Available: {list(self.catalog.garments.keys())[:5]}",
            }
        
        available_sizes = garment.available_sizes()
        
        # Get thumbnail (first available photo)
        thumbnail = None
        for size in available_sizes:
            photos = garment.get_photos_for_size(size)
            if photos:
                thumbnail = photos[0].get("image_path")
                break
        
        return {
            "status": "ready",
            "garment": {
                "id": product_id,
                "brand": brand_id,
                "name": garment.name,
                "type": garment.garment_type,
                "available_sizes": available_sizes,
                "total_reference_photos": sum(
                    len(s.photos) for s in garment.sizes.values()
                ),
                "thumbnail": thumbnail,
            },
            "requires": {
                "customer_photo": True,
                "measurements": ["bust_cm", "waist_cm", "hips_cm", "height_cm"],
                "size_selection": True,
            },
        }
    
    # ──────────────────────────────────────────────────────────
    #  Step 2: Generate — Customer submits photo + measurements
    # ──────────────────────────────────────────────────────────
    
    def generate_tryon(
        self,
        product_id: str,
        brand_id: str,
        customer_photo: str,
        bust_cm: float,
        waist_cm: float,
        hips_cm: float,
        height_cm: float = 170,
        selected_size: str = "M",
    ) -> Dict:
        """
        Generate a size-aware try-on image.
        
        Called when customer submits their photo + measurements + selected size.
        
        Returns:
            {
                "status": "success",
                "tryon_image": "path/to/output.png",
                "fit_info": { ... },
                "body_match": { ... },
                "size_alternatives": [ ... ],
            }
        """
        full_id = f"{brand_id}_{product_id}"
        garment = self.catalog.garments.get(full_id)
        
        if not garment:
            return {"status": "error", "message": "Product not found"}
        
        if not os.path.exists(customer_photo):
            return {"status": "error", "message": "Customer photo not found"}
        
        start_time = time.time()
        
        # 1. Match customer's body to closest model
        body_match = self.catalog.match_customer(
            bust_cm, waist_cm, hips_cm, height_cm, brand_id
        )
        
        # 2. Get reference photo for the selected size
        reference = self.catalog.get_reference_photo(
            full_id, selected_size,
            preferred_model_id=body_match.get("model_id")
        )
        
        # 3. Compute fit profile (physics parameters)
        fit = classify_fit(garment.garment_type, selected_size, body_match["model_size"])
        
        # 4. Load images
        customer_img = Image.open(customer_photo).convert("RGB")
        
        def get_valid_path(ref):
            if not ref or not ref.get("image_path"): return None
            # Fix windows backslashes if present in catalog json
            clean_path = ref["image_path"].replace("\\", "/")
            # Handle absolute/relative safely
            full_path = Path("/workspace/FitFusion") / clean_path
            return str(full_path) if full_path.exists() else None

        ref_path = get_valid_path(reference)
        
        if ref_path:
            garment_img = Image.open(ref_path).convert("RGB")
            source = "catalog_reference"
        else:
            # Fallback: no reference photo for this size
            # Use any available photo and apply physics resizing
            any_ref = self.catalog.get_reference_photo(full_id, "M")
            any_ref_path = get_valid_path(any_ref)
            if any_ref_path:
                garment_img = Image.open(any_ref_path).convert("RGB")
            else:
                # One last try to find ANY photo just in case M is not available
                avail = garment.available_sizes()
                last_ref_path = None
                if avail:
                    last_ref = self.catalog.get_reference_photo(full_id, avail[0])
                    last_ref_path = get_valid_path(last_ref)
                
                if last_ref_path:
                    garment_img = Image.open(last_ref_path).convert("RGB")
                else:
                    return {"status": "error", "message": f"No garment images available. Looked for {reference.get('image_path', '') if reference else 'none'}"}
            source = "physics_generated"
        
        # 5. Apply physics-based preprocessing
        prepared = self.physics.prepare_size_aware_inputs(
            person_image=customer_img,
            garment_image=garment_img,
            person_size=body_match["model_size"],
            garment_size=selected_size,
            garment_type=garment.garment_type,
        )
        
        # 6. Save preprocessed inputs (ready for IDM-VTON inference)
        session_id = f"{brand_id}_{product_id}_{selected_size}_{int(time.time())}"
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        prepared["garment_image"].save(str(session_dir / "garment_preprocessed.png"))
        prepared["agnostic_mask"].save(str(session_dir / "mask_adapted.png"))
        prepared["person_image"].save(str(session_dir / "person_resized.png"))
        
        # Save session metadata
        session_data = {
            "product_id": product_id,
            "brand_id": brand_id,
            "selected_size": selected_size,
            "customer_measurements": {
                "bust_cm": bust_cm,
                "waist_cm": waist_cm,
                "hips_cm": hips_cm,
                "height_cm": height_cm,
            },
            "body_match": body_match,
            "fit_profile": {
                "fit_type": fit.fit_type.name,
                "width_ratio": fit.width_ratio,
                "length_ratio": fit.length_ratio,
                "warp_intensity": fit.warp_intensity,
                "fabric_tension": fit.fabric_tension,
                "mask_expansion_px": fit.mask_expansion_px,
                "inpainting_strength": fit.inpainting_strength,
            },
            "prompts": {
                "positive": prepared["positive_prompt"],
                "negative": prepared["negative_prompt"],
            },
            "source": source,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }
        
        with open(session_dir / "session.json", "w") as f:
            json.dump(session_data, f, indent=2)
        
        # 7. Build size alternatives for the toggle buttons
        size_alternatives = []
        for s in garment.available_sizes():
            alt_fit = classify_fit(garment.garment_type, s, body_match["model_size"])
            alt_ratio = compute_size_ratio(garment.garment_type, s, body_match["model_size"])
            size_alternatives.append({
                "size": s,
                "fit_type": alt_fit.fit_type.name,
                "width_ratio": alt_ratio["width_ratio"],
                "description": _fit_description(alt_fit.fit_type.name),
            })
        
        elapsed = int((time.time() - start_time) * 1000)
        
        response = {
            "status": "success",
            "session_id": session_id,
            "session_dir": str(session_dir),
            "preprocessing_done": True,
            "ready_for_inference": True,
            "garment_name": garment.name,
            "selected_size": selected_size,
            "fit_info": {
                "fit_type": fit.fit_type.name,
                "description": _fit_description(fit.fit_type.name),
                "width_ratio": f"{fit.width_ratio:.2f}x",
                "fabric_feel": "tight" if fit.fabric_tension > 0.6 else 
                               "comfortable" if fit.fabric_tension > 0.3 else "loose",
            },
            "body_match": {
                "model_name": body_match["model_name"],
                "model_size": body_match["model_size"],
                "similarity": f"{body_match['similarity']}%",
            },
            "size_alternatives": size_alternatives,
            "processing_time_ms": elapsed,
            "files": {
                "garment_preprocessed": str(session_dir / "garment_preprocessed.png"),
                "mask_adapted": str(session_dir / "mask_adapted.png"),
                "person_resized": str(session_dir / "person_resized.png"),
                "prompts": {
                    "positive": prepared["positive_prompt"][:100] + "...",
                    "negative": prepared["negative_prompt"][:100] + "...",
                },
            },
            "next_step": "Run IDM-VTON inference with the preprocessed files in session_dir",
        }
        
        if garment.has_text:
            response["warning"] = "Warning: Graphic text may exhibit slight distortion due to 3D fabric wrapping simulations."
            
        return response
    
    # ──────────────────────────────────────────────────────────
    #  Quick Size Toggle (No re-upload needed)
    # ──────────────────────────────────────────────────────────
    
    def toggle_size(
        self,
        product_id: str,
        brand_id: str,
        customer_photo: str,
        bust_cm: float,
        waist_cm: float,
        hips_cm: float,
        height_cm: float,
        new_size: str,
    ) -> Dict:
        """
        Quick size toggle — customer clicks a different size button.
        
        Same as generate_tryon but optimized for speed:
        - Re-uses the customer photo (no re-upload)
        - Re-uses body match (no re-computation)
        - Only recomputes physics for the new size
        """
        return self.generate_tryon(
            product_id=product_id,
            brand_id=brand_id,
            customer_photo=customer_photo,
            bust_cm=bust_cm,
            waist_cm=waist_cm,
            hips_cm=hips_cm,
            height_cm=height_cm,
            selected_size=new_size,
        )


def _fit_description(fit_type_name: str) -> str:
    """Human-readable fit description for the UI."""
    descriptions = {
        "VERY_TIGHT": "Very tight — fabric will stretch and pull across your body",
        "TIGHT": "Tight — snug body-hugging fit, shows body contour",
        "SNUG": "Snug — close to body but comfortable, no excess fabric",
        "STANDARD": "Standard fit — comfortable and well-proportioned",
        "RELAXED": "Relaxed — slightly loose with natural room",
        "LOOSE": "Loose — noticeable room, fabric drapes away from body",
        "OVERSIZED": "Oversized — significant excess fabric, casual baggy look",
    }
    return descriptions.get(fit_type_name, "Standard fit")


# ════════════════════════════════════════════════════════════════
#  Self-Test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  FitFusion — Try-On API Self-Test")
    print("=" * 60)
    
    # Initialize with existing catalog
    api = TryOnAPI(catalog_dir="data/brand_catalog")
    
    # First, make sure we have brands onboarded
    if not api.catalog.garments:
        print("\n[0] No catalog found. Auto-onboarding from expanded dataset...")
        api.catalog.onboard_from_expanded_dataset("data/expanded_dataset")
    
    stats = api.catalog.stats()
    print(f"\nCatalog: {stats['brands']} brands, {stats['garments']} garments, "
          f"{stats['total_photos']} photos")
    
    # Step 1: Customer clicks "Try On" on a product
    garment_keys = list(api.catalog.garments.keys())
    if not garment_keys:
        print("No garments available for testing.")
        exit()
    
    test_key = garment_keys[0]
    parts = test_key.split("_", 1)
    brand_id = parts[0] if len(parts) > 1 else "unknown"
    product_id = parts[1] if len(parts) > 1 else test_key
    
    # Sometimes the brand_id has multiple underscores
    for bid in api.catalog.brands:
        if test_key.startswith(bid + "_"):
            brand_id = bid
            product_id = test_key[len(bid) + 1:]
            break
    
    print(f"\n[1] Customer clicks 'Try On' on: {product_id} (brand: {brand_id})")
    init_result = api.init_tryon(product_id, brand_id)
    
    if init_result["status"] == "ready":
        g = init_result["garment"]
        print(f"    Garment: {g['name']}")
        print(f"    Type: {g['type']}")
        print(f"    Sizes: {g['available_sizes']}")
        print(f"    Reference photos: {g['total_reference_photos']}")
    else:
        print(f"    Error: {init_result.get('message')}")
        # Try a different garment
        for key in garment_keys[:5]:
            print(f"    Available: {key}")
    
    # Step 2: Customer submits measurements (simulate — no real photo)
    print(f"\n[2] Customer selects sizes to compare:")
    
    if init_result["status"] == "ready":
        # Create a dummy customer photo for testing
        import numpy as np
        dummy = Image.fromarray(np.ones((1024, 768, 3), dtype=np.uint8) * 180)
        dummy_path = str(api.output_dir / "dummy_customer.png")
        dummy.save(dummy_path)
        
        for size in init_result["garment"]["available_sizes"][:4]:
            result = api.generate_tryon(
                product_id=product_id,
                brand_id=brand_id,
                customer_photo=dummy_path,
                bust_cm=92,
                waist_cm=76,
                hips_cm=100,
                height_cm=170,
                selected_size=size,
            )
            
            if result["status"] == "success":
                fit = result["fit_info"]
                print(f"\n    Size {size}: {fit['fit_type']}")
                print(f"      → {fit['description']}")
                print(f"      → Width: {fit['width_ratio']}, Feel: {fit['fabric_feel']}")
                print(f"      → Matched to: {result['body_match']['model_name']} "
                      f"({result['body_match']['similarity']})")
                print(f"      → Processed in {result['processing_time_ms']}ms")
    
    print("\n\n✓ API self-test complete!")
