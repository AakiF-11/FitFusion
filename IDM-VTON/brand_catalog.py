"""
FitFusion — Dynamic Brand Catalog System
==========================================
A B2B system where brand owners onboard their garments with
photos across sizes, and customers get size-aware try-ons
using real reference photos + IDM-VTON.

Brand Owner Flow:
    1. Upload garment photos on models (e.g. same dress on S, M, L, XL models)
    2. Provide garment size chart (measurements per size)
    3. Provide model measurements (bust, waist, hips, height per model)
    4. System builds a GarmentProfile with fabric behavior per size

Customer Flow:
    1. Enter body measurements (or select body type)
    2. System matches to closest model in catalog
    3. Shows real reference photo of garment on matched model
    4. Customer changes size → system swaps to correct size reference
    5. Customer wants it on their body → IDM-VTON transfer

Usage:
    from brand_catalog import BrandCatalog
    
    catalog = BrandCatalog("data/brands")
    catalog.onboard_brand("snag_tights", garments_dir="images/", 
                           size_chart=chart, models=models)
    
    match = catalog.match_customer(bust=92, waist=76, hips=98)
    reference = catalog.get_reference("borg-aviator", size="XL", model=match)
"""

import json
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import shutil


# ════════════════════════════════════════════════════════════════
#  Data Models
# ════════════════════════════════════════════════════════════════

@dataclass
class ModelProfile:
    """A model who wears the garments for photography."""
    model_id: str               # e.g. "yasmine"
    name: str                   # e.g. "Yasmine"
    bust_cm: float
    waist_cm: float
    hips_cm: float
    height_cm: float
    size_label: str             # e.g. "M", "XL"
    
    def measurement_vector(self) -> np.ndarray:
        """Body measurements as a numpy vector for distance computation."""
        return np.array([self.bust_cm, self.waist_cm, self.hips_cm, self.height_cm])


@dataclass
class GarmentSize:
    """A specific size entry for a garment."""
    size_label: str             # e.g. "S", "M", "XL"
    # Garment measurements
    chest_cm: float = 0
    waist_cm: float = 0
    hip_cm: float = 0
    length_cm: float = 0
    # Photos showing this size on models
    photos: List[Dict] = field(default_factory=list)
    # photos = [{"model_id": "yasmine", "image_path": "...", "shot": "front"}]


@dataclass
class GarmentProfile:
    """
    A garment's complete profile including all sizes and reference photos.
    This is what the brand owner creates when onboarding a garment.
    """
    garment_id: str             # e.g. "borg-aviator-black"
    brand_id: str               # e.g. "snag_tights"
    name: str                   # e.g. "Borg Aviator Jacket - Black"
    garment_type: str           # e.g. "jacket", "top", "pants", "dress"
    has_text: bool = False      # true if garment has graphic text
    worn_tucked: bool = False   # e.g. True if this top is tucked into pants/skirts
    neckline_type: str = "crew" # e.g. "v-neck", "crew", "scoop"
    fabric_rigidity: float = 0.5 # e.g. Denim = 0.9, Cotton = 0.4, Silk = 0.1
    sizes: Dict[str, GarmentSize] = field(default_factory=dict)
    # Key = size label ("S", "M", etc.)
    
    def available_sizes(self) -> List[str]:
        """List sizes that have at least one photo."""
        return [s for s, data in self.sizes.items() if data.photos]
    
    def get_photos_for_size(self, size: str) -> List[Dict]:
        """Get all photos for a specific size."""
        if size in self.sizes:
            return self.sizes[size].photos
        return []
    
    def get_photo_for_model(self, size: str, model_id: str) -> Optional[str]:
        """Get image path for a specific size and model combination."""
        for photo in self.get_photos_for_size(size):
            if photo.get("model_id") == model_id:
                return photo.get("image_path")
        return None


# ════════════════════════════════════════════════════════════════
#  Brand Catalog — The Main System
# ════════════════════════════════════════════════════════════════

class BrandCatalog:
    """
    Dynamic brand catalog system.
    
    Manages onboarding of brands, garments, and models.
    Provides body matching and reference photo selection for customers.
    """
    
    def __init__(self, catalog_dir: str = "data/brand_catalog"):
        self.catalog_dir = Path(catalog_dir)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory stores
        self.brands: Dict[str, dict] = {}
        self.models: Dict[str, ModelProfile] = {}
        self.garments: Dict[str, GarmentProfile] = {}
        
        # Load existing catalog
        self._load_catalog()
    
    # ──────────────────────────────────────────────────────────
    #  Brand Onboarding
    # ──────────────────────────────────────────────────────────
    
    def onboard_brand(
        self,
        brand_id: str,
        brand_name: str,
        models: List[Dict],
        garments: List[Dict],
        images_source_dir: str = None,
    ) -> dict:
        """
        Onboard a brand with their models and garments.
        
        Args:
            brand_id: unique id e.g. "snag_tights"
            brand_name: display name e.g. "Snag Tights"
            models: list of model dicts:
                [{"model_id": "yasmine", "name": "Yasmine", 
                  "bust_cm": 90, "waist_cm": 74, "hips_cm": 96, 
                  "height_cm": 170, "size_label": "M"}, ...]
            garments: list of garment dicts:
                [{"garment_id": "borg-aviator", "name": "Borg Aviator",
                  "garment_type": "jacket",
                  "sizes": {
                      "S": {"chest_cm": 96, "photos": [
                          {"model_id": "yasmine", "image": "img1.jpg"}
                      ]},
                      ...
                  }}, ...]
            images_source_dir: directory containing the garment images
        
        Returns:
            Summary of what was onboarded
        """
        # Create brand directory
        brand_dir = self.catalog_dir / brand_id
        brand_dir.mkdir(exist_ok=True)
        (brand_dir / "images").mkdir(exist_ok=True)
        
        # Register brand
        self.brands[brand_id] = {
            "brand_id": brand_id,
            "name": brand_name,
            "model_count": len(models),
            "garment_count": len(garments),
        }
        
        # Register models
        for m in models:
            model = ModelProfile(
                model_id=m["model_id"],
                name=m.get("name", m["model_id"]),
                bust_cm=m.get("bust_cm", 90),
                waist_cm=m.get("waist_cm", 70),
                hips_cm=m.get("hips_cm", 95),
                height_cm=m.get("height_cm", 170),
                size_label=m.get("size_label", "M"),
            )
            self.models[f"{brand_id}_{m['model_id']}"] = model
        
        # Register garments with their photos
        for g in garments:
            garment_id = g["garment_id"]
            full_id = f"{brand_id}_{garment_id}"
            
            sizes = {}
            for size_label, size_data in g.get("sizes", {}).items():
                photos = []
                for photo in size_data.get("photos", []):
                    # Copy image to catalog directory
                    src_img = photo.get("image", "")
                    if images_source_dir and src_img:
                        src_path = Path(images_source_dir) / src_img
                        if src_path.exists():
                            dest_path = brand_dir / "images" / src_img
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            if not dest_path.exists():
                                shutil.copy2(str(src_path), str(dest_path))
                            photos.append({
                                "model_id": photo.get("model_id", "unknown"),
                                "image_path": str(dest_path),
                                "shot": photo.get("shot", "front"),
                            })
                        else:
                            # Image path is already absolute or relative
                            photos.append({
                                "model_id": photo.get("model_id", "unknown"),
                                "image_path": src_img,
                                "shot": photo.get("shot", "front"),
                            })
                    else:
                        photos.append({
                            "model_id": photo.get("model_id", "unknown"),
                            "image_path": src_img,
                            "shot": photo.get("shot", "front"),
                        })
                
                sizes[size_label] = GarmentSize(
                    size_label=size_label,
                    chest_cm=size_data.get("chest_cm", 0),
                    waist_cm=size_data.get("waist_cm", 0),
                    hip_cm=size_data.get("hip_cm", 0),
                    length_cm=size_data.get("length_cm", 0),
                    photos=photos,
                )
            
            self.garments[full_id] = GarmentProfile(
                garment_id=garment_id,
                brand_id=brand_id,
                name=g.get("name", garment_id),
                garment_type=g.get("garment_type", "top"),
                has_text=g.get("has_text", False),
                worn_tucked=g.get("worn_tucked", False),
                neckline_type=g.get("neckline_type", "crew"),
                fabric_rigidity=float(g.get("fabric_rigidity", 0.5)),
                sizes=sizes,
            )
        
        # Save catalog to disk
        self._save_catalog()
        
        return {
            "brand": brand_name,
            "models_registered": len(models),
            "garments_registered": len(garments),
            "total_photos": sum(
                len(size.photos)
                for g in self.garments.values()
                if g.brand_id == brand_id
                for size in g.sizes.values()
            ),
        }
    
    def onboard_from_expanded_dataset(self, dataset_dir: str = "data/expanded_dataset"):
        """
        Auto-onboard brands from our previously expanded dataset.
        Reads pairs.txt and measurements.json to build the catalog.
        """
        dataset_path = Path(dataset_dir)
        measurements_file = dataset_path / "measurements.json"
        
        if not measurements_file.exists():
            print(f"No measurements.json found in {dataset_dir}")
            return
        
        with open(measurements_file) as f:
            measurements = json.load(f)
        
        # Group images by brand and garment
        snag_garments = {}
        us_garments = {}
        
        for filename, meas in measurements.items():
            if filename.startswith("snag_"):
                # Parse: snag_{handle}_{size}_{model}_{shot}.jpg
                parts = filename.replace(".jpg", "").split("_")
                if len(parts) >= 4:
                    handle = parts[1]
                    size_label = meas.get("size_label", "M")
                    original_size = meas.get("original_size", size_label)
                    
                    if handle not in snag_garments:
                        snag_garments[handle] = {"sizes": {}}
                    
                    if original_size not in snag_garments[handle]["sizes"]:
                        snag_garments[handle]["sizes"][original_size] = {"photos": []}
                    
                    snag_garments[handle]["sizes"][original_size]["photos"].append({
                        "model_id": parts[3] if len(parts) > 3 else "unknown",
                        "image": str(dataset_path / "image" / filename),
                    })
                    
            elif filename.startswith("us_") and not filename.startswith("us_ext_"):
                # Parse: us_{group}_{size}.jpg
                parts = filename.replace(".jpg", "").split("_")
                if len(parts) >= 3:
                    group = parts[1]
                    size_label = parts[2]
                    
                    if group not in us_garments:
                        us_garments[group] = {"sizes": {}}
                    
                    if size_label not in us_garments[group]["sizes"]:
                        us_garments[group]["sizes"][size_label] = {"photos": []}
                    
                    us_garments[group]["sizes"][size_label]["photos"].append({
                        "model_id": f"us_model_{size_label}",
                        "image": str(dataset_path / "image" / filename),
                    })
        
        # Onboard Snag Tights
        if snag_garments:
            snag_models = [
                {"model_id": "model_A", "name": "Size A", "bust_cm": 76, "waist_cm": 60, "hips_cm": 84, "height_cm": 157, "size_label": "XS"},
                {"model_id": "model_B", "name": "Size B", "bust_cm": 84, "waist_cm": 68, "hips_cm": 92, "height_cm": 162, "size_label": "S"},
                {"model_id": "model_C", "name": "Size C", "bust_cm": 92, "waist_cm": 76, "hips_cm": 100, "height_cm": 167, "size_label": "M"},
                {"model_id": "model_D", "name": "Size D", "bust_cm": 100, "waist_cm": 84, "hips_cm": 108, "height_cm": 170, "size_label": "L"},
                {"model_id": "model_E", "name": "Size E", "bust_cm": 110, "waist_cm": 94, "hips_cm": 118, "height_cm": 173, "size_label": "XL"},
                {"model_id": "model_F", "name": "Size F", "bust_cm": 120, "waist_cm": 104, "hips_cm": 128, "height_cm": 175, "size_label": "2XL"},
                {"model_id": "model_G", "name": "Size G", "bust_cm": 130, "waist_cm": 114, "hips_cm": 138, "height_cm": 175, "size_label": "3XL"},
                {"model_id": "model_H", "name": "Size H", "bust_cm": 140, "waist_cm": 124, "hips_cm": 148, "height_cm": 175, "size_label": "4XL"},
            ]
            
            garment_list = []
            for handle, data in snag_garments.items():
                garment_list.append({
                    "garment_id": handle,
                    "name": handle.replace("-", " ").title(),
                    "garment_type": "tights" if "tight" in handle else "top",
                    "sizes": data["sizes"],
                })
            
            result = self.onboard_brand(
                brand_id="snag_tights",
                brand_name="Snag Tights",
                models=snag_models,
                garments=garment_list,
            )
            print(f"Snag Tights onboarded: {result}")
        
        # Onboard Universal Standard
        if us_garments:
            us_models = [
                {"model_id": "us_model_XS", "name": "US XS Model", "bust_cm": 84, "waist_cm": 66, "hips_cm": 90, "height_cm": 165, "size_label": "XS"},
                {"model_id": "us_model_S", "name": "US S Model", "bust_cm": 90, "waist_cm": 72, "hips_cm": 96, "height_cm": 168, "size_label": "S"},
                {"model_id": "us_model_M", "name": "US M Model", "bust_cm": 96, "waist_cm": 78, "hips_cm": 102, "height_cm": 170, "size_label": "M"},
                {"model_id": "us_model_L", "name": "US L Model", "bust_cm": 104, "waist_cm": 86, "hips_cm": 110, "height_cm": 172, "size_label": "L"},
                {"model_id": "us_model_XL", "name": "US XL Model", "bust_cm": 112, "waist_cm": 94, "hips_cm": 118, "height_cm": 173, "size_label": "XL"},
                {"model_id": "us_model_2XL", "name": "US 2XL Model", "bust_cm": 120, "waist_cm": 102, "hips_cm": 128, "height_cm": 175, "size_label": "2XL"},
                {"model_id": "us_model_3XL", "name": "US 3XL Model", "bust_cm": 130, "waist_cm": 112, "hips_cm": 138, "height_cm": 175, "size_label": "3XL"},
                {"model_id": "us_model_4XL", "name": "US 4XL Model", "bust_cm": 140, "waist_cm": 122, "hips_cm": 148, "height_cm": 175, "size_label": "4XL"},
            ]
            
            garment_list = []
            for group, data in us_garments.items():
                gtype = "dress" if "dress" in group else "pants" if "pant" in group else "top"
                garment_list.append({
                    "garment_id": group,
                    "name": group.replace("-", " ").title(),
                    "garment_type": gtype,
                    "sizes": data["sizes"],
                })
            
            result = self.onboard_brand(
                brand_id="universal_standard",
                brand_name="Universal Standard",
                models=us_models,
                garments=garment_list,
            )
            print(f"Universal Standard onboarded: {result}")
    
    # ──────────────────────────────────────────────────────────
    #  Customer Body Matching
    # ──────────────────────────────────────────────────────────
    
    def match_customer(
        self,
        bust_cm: float,
        waist_cm: float,
        hips_cm: float,
        height_cm: float = 170,
        brand_id: str = None,
    ) -> Dict:
        """
        Match a customer's body measurements to the closest model.
        
        Uses a two-step process:
        1. Apply a hard filter: retain models where abs(model_height - height) <= 5.
        2. Euclidean distance strictly on (Bust, Waist, Hips) weighting geometry.
        
        Returns:
            Dict with matched model info and similarity score
        """
        customer_vec = np.array([bust_cm, waist_cm, hips_cm])
        
        # Weights: bust and hips matter most for visual appearance (no height vector)
        weights = np.array([1.5, 1.0, 1.5])
        
        best_match = None
        best_distance = float("inf")
        
        # 1. Preliminary hard filter subset (<= 5cm)
        subset = []
        for model_key, model in self.models.items():
            if brand_id and not model_key.startswith(brand_id):
                continue
            if abs(model.height_cm - height_cm) <= 5:
                subset.append(model)
                
        # 2. If the subset is empty, expand the threshold to 10cm
        if not subset:
            for model_key, model in self.models.items():
                if brand_id and not model_key.startswith(brand_id):
                    continue
                if abs(model.height_cm - height_cm) <= 10:
                    subset.append(model)
                    
        # Failsafe if completely empty
        if not subset:
            for model_key, model in self.models.items():
                if brand_id and not model_key.startswith(brand_id):
                    continue
                subset.append(model)
                
        # 3. Run the standard Euclidean distance calculation strictly on (Bust, Waist, Hips) against the pre-filtered subset
        for model in subset:
            model_vec = np.array([model.bust_cm, model.waist_cm, model.hips_cm])
            distance = np.sqrt(np.sum(weights * (customer_vec - model_vec) ** 2))
            
            if distance < best_distance:
                best_distance = distance
                best_match = model
        
        if best_match is None:
            return {"error": "No models found in catalog"}
        
        # Compute similarity score (0-100%)
        max_distance = 100.0  # max expected distance
        similarity = max(0, 100 * (1 - best_distance / max_distance))
        
        return {
            "model_id": best_match.model_id,
            "model_name": best_match.name,
            "model_size": best_match.size_label,
            "model_bust": best_match.bust_cm,
            "model_waist": best_match.waist_cm,
            "model_hips": best_match.hips_cm,
            "model_height": best_match.height_cm,
            "similarity": round(similarity, 1),
            "distance": round(best_distance, 2),
        }
    
    def get_all_matching_models(
        self,
        bust_cm: float,
        waist_cm: float,
        hips_cm: float,
        height_cm: float = 170,
        brand_id: str = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Get top-K matching models ranked by similarity.
        Useful for showing customers "models like you" to choose from.
        """
        customer_vec = np.array([bust_cm, waist_cm, hips_cm])
        weights = np.array([1.5, 1.0, 1.5])
        
        subset = []
        for model_key, model in self.models.items():
            if brand_id and not model_key.startswith(brand_id):
                continue
            if abs(model.height_cm - height_cm) <= 5:
                subset.append(model)
                
        if not subset:
            for model_key, model in self.models.items():
                if brand_id and not model_key.startswith(brand_id):
                    continue
                if abs(model.height_cm - height_cm) <= 10:
                    subset.append(model)
                    
        if not subset:
            for model_key, model in self.models.items():
                if brand_id and not model_key.startswith(brand_id):
                    continue
                subset.append(model)
        
        scores = []
        for model in subset:
            model_vec = np.array([model.bust_cm, model.waist_cm, model.hips_cm])
            distance = np.sqrt(np.sum(weights * (customer_vec - model_vec) ** 2))
            similarity = max(0, 100 * (1 - distance / 100.0))
            
            scores.append({
                "model_id": model.model_id,
                "model_name": model.name,
                "model_size": model.size_label,
                "similarity": round(similarity, 1),
                "bust": model.bust_cm,
                "waist": model.waist_cm,
                "hips": model.hips_cm,
            })
        
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]
    
    # ──────────────────────────────────────────────────────────
    #  Reference Photo Selection
    # ──────────────────────────────────────────────────────────
    
    def get_reference_photo(
        self,
        garment_full_id: str,
        size_label: str,
        preferred_model_id: str = None,
    ) -> Optional[Dict]:
        """
        Get the best reference photo for a garment at a specific size.
        
        Args:
            garment_full_id: e.g. "snag_tights_borg-aviator"
            size_label: e.g. "XL"
            preferred_model_id: if set, prefer photos from this model
        
        Returns:
            Dict with image_path, model_id, size
        """
        garment = self.garments.get(garment_full_id)
        if not garment:
            return None
        
        photos = garment.get_photos_for_size(size_label)
        if not photos:
            # Try to find the closest available size using normalized sizes
            available = garment.available_sizes()
            if not available:
                return None
            
            try:
                from size_charts import normalize_size_label
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent))
                from size_charts import normalize_size_label
            
            norm_target = normalize_size_label(size_label)
            size_order = ["4XS","3XS","2XS","XS","S","M","L","XL","2XL","3XL","4XL"]
            target_idx = size_order.index(norm_target) if norm_target in size_order else 5
            
            def size_distance(s):
                norm_s = normalize_size_label(s)
                if norm_s in size_order:
                    return abs(size_order.index(norm_s) - target_idx)
                return 999
                
            closest = min(available, key=size_distance)
            
            photos = garment.get_photos_for_size(closest)
            size_label = closest
        
        if not photos:
            return None
        
        # Prefer matching model if specified
        if preferred_model_id:
            for p in photos:
                if p.get("model_id") == preferred_model_id:
                    return {**p, "size": size_label, "garment_id": garment.garment_id}
        
        # Return first available photo
        return {**photos[0], "size": size_label, "garment_id": garment.garment_id}
    
    def get_size_comparison(
        self,
        garment_full_id: str,
        sizes: List[str] = None,
    ) -> List[Dict]:
        """
        Get reference photos across multiple sizes for comparison.
        This shows the customer how the garment looks at each size.
        """
        garment = self.garments.get(garment_full_id)
        if not garment:
            return []
        
        if sizes is None:
            sizes = garment.available_sizes()
        
        results = []
        for s in sizes:
            ref = self.get_reference_photo(garment_full_id, s)
            if ref:
                results.append(ref)
        
        return results
    
    # ──────────────────────────────────────────────────────────
    #  Customer Try-On Flow (End-to-End)
    # ──────────────────────────────────────────────────────────
    
    def customer_tryon(
        self,
        customer_bust: float,
        customer_waist: float,
        customer_hips: float,
        customer_height: float,
        garment_full_id: str,
        selected_size: str,
        customer_photo_path: str = None,
        brand_id: str = None,
    ) -> Dict:
        """
        Complete customer try-on flow.
        
        Returns:
            Dict with:
                - matched_model: closest body match
                - reference_photo: real photo of garment at selected size
                - size_alternatives: photos at other sizes
                - tryon_ready: True if we can do personal try-on
                - physics_params: fallback physics params if no reference exists
        """
        # 1. Match customer to closest model
        match = self.match_customer(
            customer_bust, customer_waist, customer_hips,
            customer_height, brand_id
        )
        
        # 2. Get reference photo for the selected size
        reference = self.get_reference_photo(
            garment_full_id, selected_size,
            preferred_model_id=match.get("model_id")
        )
        
        # 3. Get size alternatives
        garment = self.garments.get(garment_full_id)
        alternatives = []
        if garment:
            for s in garment.available_sizes():
                alt = self.get_reference_photo(garment_full_id, s)
                if alt:
                    alternatives.append(alt)
        
        # 4. Determine if personal try-on is possible
        tryon_ready = (reference is not None) and (customer_photo_path is not None)
        
        # 5. Compute physics fallback params
        from size_aware_vton import classify_fit
        physics_params = None
        if garment:
            fit = classify_fit(garment.garment_type, selected_size, match.get("model_size", "M"))
            physics_params = {
                "fit_type": fit.fit_type.name,
                "width_ratio": fit.width_ratio,
                "warp_intensity": fit.warp_intensity,
                "fabric_tension": fit.fabric_tension,
                "positive_prompt": fit.positive_prompt,
                "negative_prompt": fit.negative_prompt,
            }
        
        return {
            "matched_model": match,
            "reference_photo": reference,
            "size_alternatives": alternatives,
            "tryon_ready": tryon_ready,
            "physics_params": physics_params,
        }
    
    # ──────────────────────────────────────────────────────────
    #  Catalog Persistence
    # ──────────────────────────────────────────────────────────
    
    def _save_catalog(self):
        """Save catalog to disk as JSON."""
        catalog_data = {
            "brands": self.brands,
            "models": {k: asdict(v) for k, v in self.models.items()},
            "garments": {},
        }
        
        for gid, g in self.garments.items():
            catalog_data["garments"][gid] = {
                "garment_id": g.garment_id,
                "brand_id": g.brand_id,
                "name": g.name,
                "garment_type": g.garment_type,
                "sizes": {
                    s: {
                        "size_label": sd.size_label,
                        "chest_cm": sd.chest_cm,
                        "waist_cm": sd.waist_cm,
                        "hip_cm": sd.hip_cm,
                        "length_cm": sd.length_cm,
                        "photos": sd.photos,
                    } for s, sd in g.sizes.items()
                },
            }
        
        with open(self.catalog_dir / "catalog.json", "w") as f:
            json.dump(catalog_data, f, indent=2)
    
    def _load_catalog(self):
        """Load catalog from disk."""
        catalog_file = self.catalog_dir / "catalog.json"
        if not catalog_file.exists():
            return
        
        with open(catalog_file, encoding='utf-8-sig') as f:
            data = json.load(f)
        
        self.brands = data.get("brands", {})
        
        for k, m in data.get("models", {}).items():
            self.models[k] = ModelProfile(**m)
        
        for gid, g in data.get("garments", {}).items():
            sizes = {}
            for s, sd in g.get("sizes", {}).items():
                sizes[s] = GarmentSize(
                    size_label=sd["size_label"],
                    chest_cm=sd.get("chest_cm", 0),
                    waist_cm=sd.get("waist_cm", 0),
                    hip_cm=sd.get("hip_cm", 0),
                    length_cm=sd.get("length_cm", 0),
                    photos=sd.get("photos", []),
                )
            
            self.garments[gid] = GarmentProfile(
                garment_id=g["garment_id"],
                brand_id=g["brand_id"],
                name=g["name"],
                garment_type=g["garment_type"],
                sizes=sizes,
            )
    
    # ──────────────────────────────────────────────────────────
    #  Catalog Stats
    # ──────────────────────────────────────────────────────────
    
    def stats(self) -> Dict:
        """Get catalog statistics."""
        total_photos = sum(
            len(size.photos)
            for g in self.garments.values()
            for size in g.sizes.values()
        )
        
        return {
            "brands": len(self.brands),
            "models": len(self.models),
            "garments": len(self.garments),
            "total_photos": total_photos,
            "garment_types": list(set(g.garment_type for g in self.garments.values())),
        }


# ════════════════════════════════════════════════════════════════
#  Self-Test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  FitFusion — Brand Catalog System Self-Test")
    print("=" * 60)
    
    catalog = BrandCatalog("data/brand_catalog_test")
    
    # 1. Auto-onboard from expanded dataset
    print("\n[1] Auto-onboarding from expanded dataset...")
    catalog.onboard_from_expanded_dataset("data/expanded_dataset")
    
    # 2. Stats
    stats = catalog.stats()
    print(f"\n[2] Catalog stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    # 3. Customer matching
    print("\n[3] Customer body matching...")
    
    test_customers = [
        {"bust": 86, "waist": 68, "hips": 92, "height": 165, "desc": "Small/Medium build"},
        {"bust": 110, "waist": 94, "hips": 118, "desc": "XL build"},
        {"bust": 130, "waist": 114, "hips": 138, "desc": "3XL build"},
    ]
    
    for c in test_customers:
        match = catalog.match_customer(c["bust"], c["waist"], c["hips"], c.get("height", 170))
        print(f"\n  Customer ({c['desc']}: B={c['bust']}, W={c['waist']}, H={c['hips']}):")
        print(f"    → Best match: {match['model_name']} (size {match['model_size']})")
        print(f"    → Similarity: {match['similarity']}%")
    
    # 4. Try-on flow
    print("\n[4] Customer try-on flow...")
    garment_ids = list(catalog.garments.keys())
    if garment_ids:
        test_garment = garment_ids[0]
        
        result = catalog.customer_tryon(
            customer_bust=92,
            customer_waist=76,
            customer_hips=100,
            customer_height=170,
            garment_full_id=test_garment,
            selected_size="M",
        )
        
        print(f"\n  Garment: {test_garment}")
        print(f"  Matched model: {result['matched_model']['model_name']} "
              f"({result['matched_model']['similarity']}% match)")
        if result['reference_photo']:
            print(f"  Reference photo: {result['reference_photo']['image_path'][:60]}...")
        print(f"  Size alternatives: {len(result['size_alternatives'])} available")
        if result['physics_params']:
            print(f"  Physics fallback: {result['physics_params']['fit_type']}")
    
    # Cleanup test dir
    import shutil
    shutil.rmtree("data/brand_catalog_test", ignore_errors=True)
    
    print("\n\n✓ Self-test complete!")
