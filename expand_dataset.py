"""
FitFusion — Dataset Expansion Downloader
=======================================
Downloads images from Universal Standard and Snag Tights metadata to
build a large cross-size VTON dataset.

Features:
- Reads `products.json` and `siiys_data.json` for Universal Standard
- Reads `paired_data.json` for Snag Tights
- Downloads images concurrently
- Generates `pairs.txt` and `measurements.json` ready for training
"""

import json
import os
import requests
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import urllib.parse


# Ensure output directories exist
DATA_DIR = Path("data/expanded_dataset")
IMG_DIR = DATA_DIR / "image"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def download_image(url: str, dest_path: Path) -> bool:
    """Download an image if it doesn't exist."""
    if dest_path.exists():
        return True
    
    try:
        # Some URLs start with // instead of https://
        if url.startswith("//"):
            url = "https:" + url
            
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        # print(f"Error downloading {url}: {e}")
        return False


def process_snag_tights():
    """Extract and download image pairs from Snag Tights data."""
    print("\n--- Processing Snag Tights ---")
    
    source_file = Path("training_data_extraction/snag_tights/paired_data.json")
    if not source_file.exists():
        print("Snag Tights data not found.")
        return [], {}
        
    with open(source_file, encoding='utf-8') as f:
        data = json.load(f)
        
    pairs = []
    measurements = {}
    download_queue = []
    
    print(f"Loaded {len(data)} Snag products")
    
    for prod in data:
        handle = prod.get("handle", "")
        if not handle:
            continue
            
        sizes = prod.get("sizes", {})
        
        # Need at least 2 sizes to form a cross-size pair
        if len(sizes) < 2:
            continue
            
        # Group images by size
        size_images = {}
        for size_label, size_data in sizes.items():
            imgs = size_data.get("images", [])
            for img in imgs:
                img_url = img.get("url") or img.get("src")
                if not img_url:
                    continue
                    
                model_name = img.get("model", "Unknown")
                filename = f"snag_{handle}_{size_label}_{model_name}_{img.get('shot', 1)}.jpg".replace("/", "_")
                
                download_queue.append((img_url, IMG_DIR / filename))
                
                if size_label not in size_images:
                    size_images[size_label] = []
                size_images[size_label].append(filename)
                
                # Save measurements
                meas = size_data.get("measurements", {})
                
                # Convert height string (e.g. 152-183) to average cm
                height_str = meas.get("height_cm", "")
                height_cm = 170.0
                if "-" in height_str:
                    try:
                        pts = height_str.split("-")
                        height_cm = (float(pts[0]) + float(pts[1])) / 2
                    except:
                        pass
                elif height_str:
                    try:
                        height_cm = float(height_str)
                    except:
                        pass
                
                # Assign size label mapping
                size_map = {
                    "A": "XS", "B": "S", "C": "M", "D": "L", 
                    "E": "XL", "F": "2XL", "G": "3XL", "H": "4XL"
                }
                mapped_size = "M"
                for k, v in size_map.items():
                    if size_label.startswith(k):
                        mapped_size = v
                        break
                        
                measurements[filename] = {
                    "size_label": mapped_size,
                    "height_cm": height_cm,
                    "garment_type": "tights" if "tights" in handle else "skirt" if "skirt" in handle else "top",
                    "original_size": size_label,
                    "uk_size": meas.get("uk", ""),
                    "us_size": meas.get("us", "")
                }
        
        # Create cross-size pairs
        size_keys = list(size_images.keys())
        for i in range(len(size_keys)):
            for j in range(i + 1, len(size_keys)):
                size1 = size_keys[i]
                size2 = size_keys[j]
                
                # Match images between sizes (take first image of each)
                if size_images[size1] and size_images[size2]:
                    img1 = size_images[size1][0]
                    img2 = size_images[size2][0]
                    pairs.append((img1, img2))  # Person=img1, Garment=img2
                    pairs.append((img2, img1))  # Swap roles
                    
    # Download images
    print(f"Downloading {len(download_queue)} Snag images (this may take a while)...")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_image, url, path) for url, path in download_queue]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                success += 1
                
    print(f"Downloaded {success}/{len(download_queue)} Snag images")
    print(f"Generated {len(pairs)} Snag pairs")
    
    return pairs, measurements


def process_universal_standard():
    """Extract and download image pairs from Universal Standard data."""
    print("\n--- Processing Universal Standard ---")
    
    source_file = Path("training_data_extraction/universal_standard/siiys_data.json")
    if not source_file.exists():
        print("US SIIYS data not found.")
        return [], {}
        
    with open(source_file, encoding='utf-8') as f:
        data = json.load(f)
        
    pairs = []
    measurements = {}
    download_queue = []
    
    print(f"Loaded {len(data)} US SIIYS groups")
    
    for group_id, group_data in data.items():
        sizes = group_data.get("sizes", [])
        
        # Need at least 2 sizes to form a cross-size pair
        if len(sizes) < 2:
            continue
            
        size_filenames = {}
        
        for size_info in sizes:
            size_label = size_info.get("size_label", "")
            img_url = size_info.get("image_url", "")
            if not size_label or not img_url:
                continue
                
            filename = f"us_{group_id}_{size_label}.jpg".replace("/", "_")
            download_queue.append((img_url, IMG_DIR / filename))
            size_filenames[size_label] = filename
            
            # Parse measurements
            model_info = size_info.get("model", {})
            height_str = model_info.get("height", "")
            bust_str = model_info.get("bust", "")
            waist_str = model_info.get("waist", "")
            hips_str = model_info.get("hips", "")
            
            # Height to cm: 5' 9.5"
            height_cm = 170.0
            if "'" in height_str:
                try:
                    parts = height_str.replace('"', '').split("'")
                    feet = float(parts[0].strip())
                    inches = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
                    height_cm = (feet * 12 + inches) * 2.54
                except:
                    pass
                    
            # Bust/Waist/Hips to cm
            def parse_inches(s):
                if not s: return 0.0
                try: return float(s.replace('"', '').replace("'", "").strip()) * 2.54
                except: return 0.0
                
            measurements[filename] = {
                "size_label": size_label.upper(),
                "height_cm": height_cm,
                "bust_cm": parse_inches(bust_str),
                "waist_cm": parse_inches(waist_str),
                "hips_cm": parse_inches(hips_str),
                "garment_type": "dress" if "dress" in group_id else "pants" if "pant" in group_id else "top"
            }
            
        # Create cross-size pairs
        size_keys = list(size_filenames.keys())
        for i in range(len(size_keys)):
            for j in range(i + 1, len(size_keys)):
                size1 = size_keys[i]
                size2 = size_keys[j]
                
                img1 = size_filenames[size1]
                img2 = size_filenames[size2]
                
                pairs.append((img1, img2))  # Person=img1, Garment=img2
                pairs.append((img2, img1))  # Swap
                
    # Parse the larger products.json for more images
    prod_file = Path("training_data_extraction/universal_standard/products.json")
    if prod_file.exists():
        try:
            with open(prod_file, encoding='utf-8') as f:
                prod_data = json.load(f)
                
            print(f"Loaded {len(prod_data)} US products for extra extraction")
            
            # Simple extraction: just grab the first image of each product as a self-pair
            # to increase diversity of garments
            for prod in prod_data:
                handle = prod.get("handle", "")
                imgs = prod.get("images", [])
                if not handle or not imgs:
                    continue
                    
                img_obj = imgs[0]
                img_url = ""
                if isinstance(img_obj, dict):
                    img_url = img_obj.get("url") or img_obj.get("src", "")
                elif isinstance(img_obj, str):
                    img_url = img_obj
                
                if not img_url:
                    continue
                    
                filename = f"us_ext_{handle}.jpg".replace("/", "_")
                
                # Check if it's clothing (skip accessories)
                prod_type = prod.get("product_type", "").lower()
                if prod_type in ["jewelry", "accessories", "gift card"]:
                    continue
                    
                download_queue.append((img_url, IMG_DIR / filename))
                
                # We don't have exact measurements for these, use default M
                measurements[filename] = {
                    "size_label": "M",
                    "height_cm": 170.0,
                    "garment_type": "dress" if "dress" in handle else "pants" if "pant" in handle else "top"
                }
                
                # Add as a self-pair (auto-reconstruction task)
                pairs.append((filename, filename))
                
        except Exception as e:
            print(f"Warning: Could not parse US products.json: {e}")
            
    # Download images
    print(f"Downloading {len(download_queue)} US images...")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_image, url, path) for url, path in download_queue]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                success += 1
                
    print(f"Downloaded {success}/{len(download_queue)} US images")
    print(f"Generated {len(pairs)} US pairs")
    
    return pairs, measurements


def main():
    print("Starting dataset expansion...")
    
    all_pairs = []
    all_measurements = {}
    
    # Process Snag Tights
    snag_pairs, snag_meas = process_snag_tights()
    all_pairs.extend(snag_pairs)
    all_measurements.update(snag_meas)
    
    # Process Universal Standard
    us_pairs, us_meas = process_universal_standard()
    all_pairs.extend(us_pairs)
    all_measurements.update(us_meas)
    
    print("\n--- Final Dataset Build ---")
    
    # Filter to only keep pairs where both files exist
    valid_pairs = []
    for p_img, c_img in all_pairs:
        if (IMG_DIR / p_img).exists() and (IMG_DIR / c_img).exists():
            valid_pairs.append((p_img, c_img))
            
    # Remove duplicates
    valid_pairs = list(set(valid_pairs))
    
    print(f"Total valid pairs: {len(valid_pairs)}")
    print(f"Total images with measurements: {len(all_measurements)}")
    
    # Save pairs.txt
    with open(DATA_DIR / "pairs.txt", "w") as f:
        for p, c in valid_pairs:
            f.write(f"{p} {c}\n")
            
    # Save measurements.json
    with open(DATA_DIR / "measurements.json", "w") as f:
        json.dump(all_measurements, f, indent=2)
        
    print("\n✓ Dataset expansion complete!")
    print(f"  Images saved to {IMG_DIR}")
    print(f"  Pairs saved to {DATA_DIR / 'pairs.txt'}")
    print(f"  Measurements saved to {DATA_DIR / 'measurements.json'}")


if __name__ == "__main__":
    main()
