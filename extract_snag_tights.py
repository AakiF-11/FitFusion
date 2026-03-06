"""
Snag Tights — Size-Paired Data Extractor
==========================================
Snag Tights filenames encode: ModelName-Size-ProductName_ShotNumber.jpg

Example: Taryn-Size-G2-BorgAviator-Black_5.jpg
  => Model: Taryn, Size: G2 (UK 26), Product: BorgAviator-Black, Shot: 5

Snag size guide (A-J):
  A/B = UK 4-6  |  C = UK 8-10  |  D = UK 12-14
  E = UK 16-18  |  F = UK 20-22  |  G = UK 24-26
  H = UK 28-30  |  I = UK 32-34  |  J = UK 36-38

Phase 1: Analysis — How much paired data exists?
Phase 2: Extract all paired images with metadata
Phase 3: Download images grouped by product/size
"""

import requests
import re
import json
import sys
import time
import os
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

S = requests.Session()
S.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
})

BASE_DIR = r"c:\Users\Aakif\Desktop\FitFusion\training_data_extraction\snag_tights"

# Snag size chart — body measurements per letter size
# These come from snagtights.com/pages/size-chart
SNAG_SIZE_CHART = {
    'A/B': {'uk': '4-6', 'us': '0-2', 'hip_cm': '81-91', 'waist_cm': '61-71', 'height_cm': '152-168'},
    'A': {'uk': '4-6', 'us': '0-2', 'hip_cm': '81-91', 'waist_cm': '61-71', 'height_cm': '152-168'},
    'B': {'uk': '4-6', 'us': '0-2', 'hip_cm': '81-91', 'waist_cm': '61-71', 'height_cm': '152-168'},
    'C': {'uk': '8-10', 'us': '4-6', 'hip_cm': '91-101', 'waist_cm': '71-81', 'height_cm': '152-178'},
    'C1': {'uk': '8', 'us': '4', 'hip_cm': '91-96', 'waist_cm': '71-76', 'height_cm': '152-178'},
    'C2': {'uk': '10', 'us': '6', 'hip_cm': '96-101', 'waist_cm': '76-81', 'height_cm': '152-178'},
    'D': {'uk': '12-14', 'us': '8-10', 'hip_cm': '101-112', 'waist_cm': '81-91', 'height_cm': '152-183'},
    'D1': {'uk': '12', 'us': '8', 'hip_cm': '101-106', 'waist_cm': '81-86', 'height_cm': '152-183'},
    'D2': {'uk': '14', 'us': '10', 'hip_cm': '106-112', 'waist_cm': '86-91', 'height_cm': '152-183'},
    'E': {'uk': '16-18', 'us': '12-14', 'hip_cm': '112-122', 'waist_cm': '91-101', 'height_cm': '152-183'},
    'E1': {'uk': '16', 'us': '12', 'hip_cm': '112-117', 'waist_cm': '91-96', 'height_cm': '152-183'},
    'E2': {'uk': '18', 'us': '14', 'hip_cm': '117-122', 'waist_cm': '96-101', 'height_cm': '152-183'},
    'F': {'uk': '20-22', 'us': '16-18', 'hip_cm': '122-132', 'waist_cm': '101-112', 'height_cm': '157-183'},
    'F1': {'uk': '20', 'us': '16', 'hip_cm': '122-127', 'waist_cm': '101-106', 'height_cm': '157-183'},
    'F2': {'uk': '22', 'us': '18', 'hip_cm': '127-132', 'waist_cm': '106-112', 'height_cm': '157-183'},
    'Short F': {'uk': '20-22', 'us': '16-18', 'hip_cm': '122-132', 'waist_cm': '101-112', 'height_cm': '147-157'},
    'G': {'uk': '24-26', 'us': '20-22', 'hip_cm': '132-142', 'waist_cm': '112-122', 'height_cm': '157-183'},
    'G1': {'uk': '24', 'us': '20', 'hip_cm': '132-137', 'waist_cm': '112-117', 'height_cm': '157-183'},
    'G2': {'uk': '26', 'us': '22', 'hip_cm': '137-142', 'waist_cm': '117-122', 'height_cm': '157-183'},
    'H': {'uk': '28-30', 'us': '24-26', 'hip_cm': '142-152', 'waist_cm': '122-132', 'height_cm': '157-183'},
    'H1': {'uk': '28', 'us': '24', 'hip_cm': '142-147', 'waist_cm': '122-127', 'height_cm': '157-183'},
    'H2': {'uk': '30', 'us': '26', 'hip_cm': '147-152', 'waist_cm': '127-132', 'height_cm': '157-183'},
    'I': {'uk': '32-34', 'us': '28-30', 'hip_cm': '152-162', 'waist_cm': '132-142', 'height_cm': '157-183'},
    'I1': {'uk': '32', 'us': '28', 'hip_cm': '152-157', 'waist_cm': '132-137', 'height_cm': '157-183'},
    'I2': {'uk': '34', 'us': '30', 'hip_cm': '157-162', 'waist_cm': '137-142', 'height_cm': '157-183'},
    'J': {'uk': '36-38', 'us': '32-34', 'hip_cm': '162-173', 'waist_cm': '142-152', 'height_cm': '157-183'},
    'J1': {'uk': '36', 'us': '32', 'hip_cm': '162-167', 'waist_cm': '142-147', 'height_cm': '157-183'},
    'J2': {'uk': '38', 'us': '34', 'hip_cm': '167-173', 'waist_cm': '147-152', 'height_cm': '157-183'},
}

def safe_get(url, timeout=30):
    for attempt in range(3):
        try:
            return S.get(url, timeout=timeout)
        except:
            if attempt < 2:
                time.sleep(2)
    return None

def parse_image_filename(filename):
    """
    Parse Snag Tights image filename to extract model, size, product, shot.
    
    Patterns found:
      Taryn-Size-G2-BorgAviator-Black_5.jpg
      Yasmine-Size-D1-BorgAviator-Black_5.jpg
      Goodie-Size_D1-Borg_Aviator-Black_7.jpg  (underscore variant)
      Details_4.jpg (detail shots, no model)
    """
    # Remove extension
    name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    # Pattern 1: ModelName-Size-XX-ProductName_Shot
    m = re.match(r'^([A-Za-z]+)-Size[-_]([A-Za-z0-9/]+)[-_](.+?)_(\d+)$', name)
    if m:
        return {
            'model': m.group(1),
            'size': m.group(2),
            'product_name': m.group(3),
            'shot': int(m.group(4)),
        }
    
    # Pattern 2: ModelName-SizeXX-ProductName_Shot (no dash after Size)
    m = re.match(r'^([A-Za-z]+)-Size([A-Za-z0-9/]+)-(.+?)_(\d+)$', name)
    if m:
        return {
            'model': m.group(1),
            'size': m.group(2),
            'product_name': m.group(3),
            'shot': int(m.group(4)),
        }
    
    # Pattern 3: Detail shots
    if name.lower().startswith('detail'):
        return {
            'model': None,
            'size': None,
            'product_name': 'detail',
            'shot': None,
        }
    
    # Pattern 4: Product color name (no model/size info)
    # e.g. longline-hoodie-black-snag-1.jpg
    return {
        'model': None,
        'size': None,
        'product_name': name,
        'shot': None,
    }


# ═══════════════════════════════════════════════════════════
# PHASE 1: Fetch ALL products and analyze paired data
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 1: Fetch & Analyze ALL Snag Tights Products")
print("=" * 70)

all_products = []
page = 1
while True:
    r = safe_get(f"https://www.snagtights.com/products.json?limit=250&page={page}")
    if not r or r.status_code != 200:
        break
    try:
        data = r.json()
        products = data.get('products', [])
        if not products:
            break
        all_products.extend(products)
        print(f"  Page {page}: {len(products)} products (total: {len(all_products)})")
        page += 1
        time.sleep(0.5)
    except:
        break

print(f"\n  Total products fetched: {len(all_products)}")

# ═══════════════════════════════════════════════════════════
# PHASE 2: Parse all image filenames for size-paired data
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2: Parse Image Filenames for Size-Paired Data")
print("=" * 70)

paired_products = []  # Products with multiple sizes photographed
all_models = set()
all_sizes = set()
total_paired_images = 0

for p in all_products:
    handle = p.get('handle', '')
    title = p.get('title', '')
    images = p.get('images', [])
    
    # Parse all image filenames
    parsed = []
    for img in images:
        src = img.get('src', '')
        filename = src.split('/')[-1].split('?')[0] if src else ''
        info = parse_image_filename(filename)
        info['src'] = src
        info['filename'] = filename
        parsed.append(info)
    
    # Find images with size info
    sized_images = [x for x in parsed if x['size']]
    unique_sizes = set(x['size'] for x in sized_images)
    unique_models = set(x['model'] for x in sized_images if x['model'])
    
    if len(unique_sizes) >= 2:  # Multiple sizes = paired data!
        paired_products.append({
            'handle': handle,
            'title': title,
            'total_images': len(images),
            'sized_images': len(sized_images),
            'unique_sizes': sorted(unique_sizes),
            'unique_models': sorted(unique_models),
            'images': parsed,
        })
        total_paired_images += len(sized_images)
        all_models.update(unique_models)
        all_sizes.update(unique_sizes)

print(f"\n  Products with paired size data: {len(paired_products)}")
print(f"  Total paired images: {total_paired_images}")
print(f"  Unique models: {len(all_models)} — {sorted(all_models)}")
print(f"  Unique sizes: {len(all_sizes)} — {sorted(all_sizes)}")

# Show paired data distribution
print(f"\n  Paired products by number of sizes:")
size_dist = defaultdict(int)
for pp in paired_products:
    n = len(pp['unique_sizes'])
    size_dist[n] += 1
for n in sorted(size_dist.keys()):
    print(f"    {n} sizes: {size_dist[n]} products")

# Show top paired products
print(f"\n  Top 15 paired products:")
paired_products.sort(key=lambda x: x['sized_images'], reverse=True)
for pp in paired_products[:15]:
    print(f"    {pp['title'][:45]}: {pp['sized_images']} sized imgs, "
          f"sizes={pp['unique_sizes']}, models={pp['unique_models']}")

# ═══════════════════════════════════════════════════════════
# PHASE 3: Save metadata + Download paired images
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 3: Save Metadata & Download Paired Images")
print("=" * 70)

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'paired_images'), exist_ok=True)

# Save complete product catalog
catalog = []
for p in all_products:
    catalog.append({
        'handle': p.get('handle'),
        'title': p.get('title'),
        'product_type': p.get('product_type'),
        'tags': p.get('tags', []),
        'options': [{
            'name': o.get('name'),
            'values': o.get('values', [])
        } for o in p.get('options', [])],
        'num_images': len(p.get('images', [])),
        'num_variants': len(p.get('variants', [])),
    })

with open(os.path.join(BASE_DIR, 'products.json'), 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)
print(f"  Saved products.json ({len(catalog)} products)")

# Save paired data metadata
paired_meta = []
for pp in paired_products:
    sized_imgs = [x for x in pp['images'] if x['size']]
    
    # Group by size
    by_size = defaultdict(list)
    for img in sized_imgs:
        by_size[img['size']].append({
            'model': img['model'],
            'shot': img['shot'],
            'filename': img['filename'],
            'src': img['src'],
        })
    
    paired_meta.append({
        'handle': pp['handle'],
        'title': pp['title'],
        'sizes': {
            size: {
                'count': len(imgs),
                'models': list(set(i['model'] for i in imgs if i['model'])),
                'measurements': SNAG_SIZE_CHART.get(size, {}),
                'images': imgs,
            }
            for size, imgs in sorted(by_size.items())
        },
        'total_sized_images': len(sized_imgs),
        'num_sizes': len(by_size),
    })

with open(os.path.join(BASE_DIR, 'paired_data.json'), 'w', encoding='utf-8') as f:
    json.dump(paired_meta, f, indent=2, ensure_ascii=False)
print(f"  Saved paired_data.json ({len(paired_meta)} products)")

# Save size chart
with open(os.path.join(BASE_DIR, 'size_chart.json'), 'w', encoding='utf-8') as f:
    json.dump(SNAG_SIZE_CHART, f, indent=2, ensure_ascii=False)
print(f"  Saved size_chart.json")

# Download paired images — organized by product/size
download_count = 0
error_count = 0
DOWNLOAD_LIMIT = None  # Set to e.g. 500 to limit downloads

for pp in paired_products:
    product_dir = os.path.join(BASE_DIR, 'paired_images', pp['handle'])
    os.makedirs(product_dir, exist_ok=True)
    
    sized_imgs = [x for x in pp['images'] if x['size']]
    
    for img in sized_imgs:
        if DOWNLOAD_LIMIT and download_count >= DOWNLOAD_LIMIT:
            break
        
        size = img['size']
        model = img['model'] or 'unknown'
        shot = img['shot'] or 0
        
        # Create size subdirectory
        size_dir = os.path.join(product_dir, f"size_{size}")
        os.makedirs(size_dir, exist_ok=True)
        
        # Download
        filename = f"{model}_shot{shot}.jpg"
        filepath = os.path.join(size_dir, filename)
        
        if os.path.exists(filepath):
            download_count += 1
            continue
        
        try:
            r = S.get(img['src'], timeout=15)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                download_count += 1
            else:
                error_count += 1
        except:
            error_count += 1
        
        if download_count % 50 == 0:
            print(f"    Downloaded: {download_count} | Errors: {error_count}")
            time.sleep(0.3)
        
        time.sleep(0.1)
    
    if DOWNLOAD_LIMIT and download_count >= DOWNLOAD_LIMIT:
        print(f"  Download limit reached ({DOWNLOAD_LIMIT})")
        break

print(f"\n  Total downloaded: {download_count}")
print(f"  Errors: {error_count}")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXTRACTION SUMMARY")
print("=" * 70)
print(f"  Total products: {len(all_products)}")
print(f"  Products with paired size data: {len(paired_products)}")
print(f"  Total paired images: {total_paired_images}")
print(f"  Unique models: {len(all_models)}")
print(f"  Unique sizes: {len(all_sizes)}")
print(f"  Images downloaded: {download_count}")
print(f"  Download errors: {error_count}")
print(f"  Output directory: {BASE_DIR}")
print("=" * 70)
