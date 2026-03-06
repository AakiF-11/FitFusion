"""
Universal Standard — Complete Catalog + SIIYS Extraction
=========================================================
Focused on extracting the gold-standard training data:
  - Full product catalog from Shopify API (1,855 products)
  - SIIYS (See It In Your Size) paired multi-size images
  - Model measurements (height, bust, waist, hips) per size
  - Structured output with per-group SIIYS directories

SIIYS is the KEY data: same garment photographed on 11 different body sizes
with actual measurements — this is what enables size-aware VTON training.
"""

import requests
import json
import re
import os
import sys
import time
import hashlib
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent / "training_data_extraction" / "universal_standard"
IMG_DIR = BASE_DIR / "images"
SIIYS_DIR = BASE_DIR / "siiys_images"
SIIYS_GROUPS_DIR = BASE_DIR / "siiys_groups"  # Structured per-group dirs
CHECKPOINT_FILE = BASE_DIR / "checkpoint.json"
PRODUCTS_FILE = BASE_DIR / "products.json"
SIIYS_DATA_FILE = BASE_DIR / "siiys_data.json"

DOMAIN = "https://www.universalstandard.com"
MAX_IMG_PER_PRODUCT = 8  # Regular product images

# Image resolution for SIIYS (higher = better for training)
SIIYS_IMG_WIDTH = 1000  # Override width=500 default

# Session
S = requests.Session()
S.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
})


def download_img(url, save_path, timeout=15):
    """Download image if not already exists."""
    if save_path.exists() and save_path.stat().st_size > 1000:
        return True  # Already have it
    try:
        if url.startswith('//'):
            url = 'https:' + url
        r = S.get(url, timeout=timeout, stream=True)
        if r.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return save_path.stat().st_size > 1000
    except Exception as e:
        pass
    return False


def upgrade_siiys_url(url, width=SIIYS_IMG_WIDTH):
    """Get higher resolution SIIYS image URL."""
    if url.startswith('//'):
        url = 'https:' + url
    # Replace width parameter
    url = re.sub(r'[?&]width=\d+', f'?width={width}', url)
    if '?width=' not in url and '&width=' not in url:
        url += f'?width={width}'
    return url


def save_checkpoint(data):
    """Save checkpoint for resume capability."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: FETCH ALL PRODUCTS FROM SHOPIFY
# ═══════════════════════════════════════════════════════════════════
def fetch_all_products():
    """Fetch all products from Shopify products.json API."""
    print(f"\n{'='*60}")
    print("  PHASE 1: Fetching all products from Shopify")
    print(f"{'='*60}")
    
    all_products = []
    seen_handles = set()
    page = 1
    
    while True:
        url = f"{DOMAIN}/products.json?limit=250&page={page}"
        try:
            r = S.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  Page {page}: HTTP {r.status_code} — stopping")
                break
            data = r.json()
            products = data.get('products', [])
            if not products:
                break
            
            new = 0
            for p in products:
                h = p.get('handle', '')
                if h and h not in seen_handles:
                    seen_handles.add(h)
                    all_products.append(p)
                    new += 1
            
            print(f"  Page {page}: {len(products)} products ({new} new, total: {len(all_products)})")
            page += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  Page {page} error: {e}")
            break
    
    print(f"\n  Total unique products: {len(all_products)}")
    return all_products


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: IDENTIFY SIIYS GROUPS
# ═══════════════════════════════════════════════════════════════════
def identify_siiys_groups(all_products):
    """Find all unique SIIYS groups and their products."""
    print(f"\n{'='*60}")
    print("  PHASE 2: Identifying SIIYS groups")
    print(f"{'='*60}")
    
    groups = {}  # group_name -> list of product handles
    
    for p in all_products:
        tags = p.get('tags', '')
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        else:
            tags = tags or []
        
        for t in tags:
            if t.startswith('siiys:'):
                group = t.replace('siiys:', '').strip()
                if group not in groups:
                    groups[group] = []
                groups[group].append(p['handle'])
                break
    
    print(f"\n  Unique SIIYS groups: {len(groups)}")
    for group, handles in sorted(groups.items()):
        print(f"    {group:30s} — {len(handles)} products")
    
    return groups


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: SCRAPE SIIYS IMAGES + MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════
def scrape_siiys_group(group_name, product_handle):
    """Scrape SIIYS images and measurements from one product page."""
    url = f"{DOMAIN}/products/{product_handle}"
    
    try:
        r = S.get(url, timeout=30)
        html = r.text
    except Exception as e:
        print(f"    ERROR fetching {url}: {e}")
        return None
    
    # Find SIIYS desktop section
    idx = html.find('productSiiysDesktop')
    if idx < 0:
        # Try alternate patterns
        idx = html.find('productSiiys"')
        if idx < 0:
            idx = html.find('See It In Your Size')
            if idx > 0:
                idx = max(0, idx - 1000)
    
    if idx < 0:
        return None
    
    section = html[idx:idx+100000]
    
    # Extract thumbnail images with size labels
    slides = re.findall(
        r'productSiiysThumbnailSwiperSlideImg"\s+src="([^"]+)".*?'
        r'productSiiysThumbnailSwiperSlideSize">\s*([^<]+?)\s*</div>',
        section, re.DOTALL
    )
    
    # Also get main (larger) images
    main_imgs = re.findall(
        r'productSiiysSwiperSlideImg"\s+src="([^"]+)"',
        section
    )
    
    # Extract body measurements (6 columns: Name, Size, Height, Bust, Waist, Hips)
    meas_cells = re.findall(
        r'SiiysSwiperSlideSizeGridBottomItem">\s*([^<]+?)\s*</div>',
        section
    )
    
    # Parse measurement groups (6 cells per model)
    measurements = []
    for i in range(0, len(meas_cells), 6):
        if i + 5 < len(meas_cells):
            measurements.append({
                'product_name': meas_cells[i].strip(),
                'size': meas_cells[i + 1].strip(),
                'height': meas_cells[i + 2].strip(),
                'bust': meas_cells[i + 3].strip(),
                'waist': meas_cells[i + 4].strip(),
                'hips': meas_cells[i + 5].strip(),
            })
    
    if not slides and not main_imgs:
        return None
    
    result = {
        'group': group_name,
        'source_product': product_handle,
        'num_sizes': len(slides),
        'sizes': [],
        'measurements': measurements,
    }
    
    # Create structured output directory for this group
    safe_group = re.sub(r'[^a-zA-Z0-9_-]', '_', group_name)
    group_dir = SIIYS_GROUPS_DIR / safe_group
    group_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for si, (thumb_url, size_label) in enumerate(slides):
        size_label = size_label.strip()
        
        # Get higher-res main image if available, otherwise use thumbnail
        img_url = main_imgs[si] if si < len(main_imgs) else thumb_url
        img_url = upgrade_siiys_url(img_url)
        
        # Also save thumbnail
        thumb_url_hires = upgrade_siiys_url(thumb_url, 500)
        
        safe_size = re.sub(r'[^a-zA-Z0-9_-]', '_', size_label)
        
        # Save main image
        main_fname = f"{safe_group}_size_{safe_size}.jpg"
        main_path = group_dir / main_fname
        main_ok = download_img(img_url, main_path)
        
        # Also save to flat siiys_images for backward compat
        flat_fname = f"{safe_group}_{safe_size}.jpg"
        flat_path = SIIYS_DIR / flat_fname
        download_img(img_url, flat_path)
        
        size_entry = {
            'size_label': size_label,
            'image_url': img_url,
            'local_path': str(main_path.relative_to(BASE_DIR)) if main_ok else None,
        }
        
        # Attach measurements for this size
        if si < len(measurements):
            size_entry['model'] = measurements[si]
        
        result['sizes'].append(size_entry)
        if main_ok:
            downloaded += 1
    
    # Save group metadata
    group_meta_path = group_dir / "siiys_metadata.json"
    with open(group_meta_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def extract_all_siiys(groups, existing_groups=None):
    """Extract SIIYS data for all groups."""
    print(f"\n{'='*60}")
    print("  PHASE 3: Extracting SIIYS images + measurements")
    print(f"{'='*60}")
    
    if existing_groups is None:
        existing_groups = set()
    
    all_siiys = {}
    total_images = 0
    
    for gi, (group_name, handles) in enumerate(sorted(groups.items())):
        safe_group = re.sub(r'[^a-zA-Z0-9_-]', '_', group_name)
        group_dir = SIIYS_GROUPS_DIR / safe_group
        meta_file = group_dir / "siiys_metadata.json"
        
        # Check if already complete
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    existing = json.load(f)
                if existing.get('sizes') and len(existing['sizes']) >= 8:
                    print(f"  [{gi+1}/{len(groups)}] {group_name} — already complete ({len(existing['sizes'])} sizes)")
                    all_siiys[group_name] = existing
                    total_images += len(existing['sizes'])
                    continue
            except:
                pass
        
        print(f"  [{gi+1}/{len(groups)}] {group_name} — scraping from {handles[0]}...")
        
        result = scrape_siiys_group(group_name, handles[0])
        
        if result and result['sizes']:
            all_siiys[group_name] = result
            downloaded = sum(1 for s in result['sizes'] if s.get('local_path'))
            total_images += downloaded
            print(f"    OK: {len(result['sizes'])} sizes, {downloaded} images downloaded, "
                  f"{len(result.get('measurements', []))} measurements")
        else:
            # Try other products in the same group
            found = False
            for alt_handle in handles[1:3]:  # Try up to 2 alternatives
                print(f"    Trying alternative: {alt_handle}...")
                result = scrape_siiys_group(group_name, alt_handle)
                if result and result['sizes']:
                    all_siiys[group_name] = result
                    downloaded = sum(1 for s in result['sizes'] if s.get('local_path'))
                    total_images += downloaded
                    print(f"    OK: {len(result['sizes'])} sizes, {downloaded} images")
                    found = True
                    break
                time.sleep(0.5)
            
            if not found:
                print(f"    MISS: No SIIYS images found for group {group_name}")
        
        time.sleep(0.5)
    
    print(f"\n  SIIYS extraction complete:")
    print(f"    Groups with data: {len(all_siiys)}/{len(groups)}")
    print(f"    Total size images: {total_images}")
    
    return all_siiys


# ═══════════════════════════════════════════════════════════════════
# PHASE 4: PROCESS PRODUCT CATALOG
# ═══════════════════════════════════════════════════════════════════
def process_products(all_products, siiys_data):
    """Process all products and download product images."""
    print(f"\n{'='*60}")
    print("  PHASE 4: Processing product catalog + downloading images")
    print(f"{'='*60}")
    
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    start_idx = 0
    extracted = []
    
    if checkpoint and checkpoint.get('phase') == 'products':
        start_idx = checkpoint.get('last_idx', 0) + 1
        extracted = checkpoint.get('products', [])
        print(f"  Resuming from product {start_idx} ({len(extracted)} already done)")
    
    img_count = 0
    skipped_imgs = 0
    
    for i in range(start_idx, len(all_products)):
        raw = all_products[i]
        handle = raw.get('handle', '')
        title = raw.get('title', '')
        body_html = raw.get('body_html', '') or ''
        tags_raw = raw.get('tags', '')
        
        if i % 25 == 0:
            print(f"  [{i+1}/{len(all_products)}] {title[:50]}...")
        
        # Parse tags
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
        else:
            tags = tags_raw or []
        
        # Check SIIYS group
        siiys_group = ''
        for t in tags:
            if t.startswith('siiys:'):
                siiys_group = t.replace('siiys:', '').strip()
                break
        
        # Clean HTML
        body_clean = re.sub(r'<[^>]+>', ' ', body_html)
        body_clean = re.sub(r'\s+', ' ', body_clean).strip()
        
        # Parse variants / sizes
        variants = raw.get('variants', [])
        sizes = []
        for v in variants:
            sizes.append({
                'name': v.get('title', ''),
                'option1': v.get('option1', ''),
                'option2': v.get('option2', ''),
                'sku': v.get('sku', ''),
                'price': v.get('price', ''),
                'available': v.get('available', False),
                'weight_grams': v.get('grams', 0),
            })
        
        # Extract model info from body
        model_height = ''
        model_size = ''
        m_match = re.search(
            r"[Mm]odel[:\s]+(\d+[''′]\s*\d+[\"\"″]?)\s+wearing\s+(\S+)",
            body_clean
        )
        if m_match:
            model_height = m_match.group(1)
            model_size = m_match.group(2)
        else:
            m_match2 = re.search(
                r"(\d+[''′]\s*\d+[\"\"″]?)\s+wearing\s+(\S+)",
                body_clean
            )
            if m_match2:
                model_height = m_match2.group(1)
                model_size = m_match2.group(2)
        
        # Extract fabric
        fabric = ''
        for pat in [r'(\d+%\s*\w+(?:[\s,/]+\d+%\s*\w+)*)', r'[Ff]abric[:\s]+([^.]+)']:
            fm = re.search(pat, body_clean)
            if fm:
                fabric = fm.group(1).strip()
                break
        
        # Fit type
        fit_type = ''
        for fit in ['slim fit', 'regular fit', 'relaxed fit', 'oversized', 'loose fit',
                     'skinny', 'straight leg', 'wide leg', 'bootcut', 'tapered', 'fitted']:
            if fit in body_clean.lower() or fit in title.lower():
                fit_type = fit
                break
        
        # Product type categorization
        product_type = raw.get('product_type', '')
        
        product = {
            'brand': 'Universal Standard',
            'product_id': str(raw.get('id', '')),
            'handle': handle,
            'name': title,
            'product_type': product_type,
            'tags': tags,
            'siiys_group': siiys_group,
            'description': body_clean[:500],
            'price': variants[0].get('price', '') if variants else '',
            'currency': 'USD',
            'sizes': sizes,
            'size_range': f"{sizes[0]['option1'] if sizes else ''} - {sizes[-1]['option1'] if sizes else ''}",
            'fit_type': fit_type,
            'fabric': fabric,
            'model_height': model_height,
            'model_size_worn': model_size,
            'images': [],
            'siiys_group_data': None,
        }
        
        # Link SIIYS data if this product belongs to a SIIYS group
        if siiys_group and siiys_group in siiys_data:
            product['siiys_group_data'] = {
                'group': siiys_group,
                'num_sizes': siiys_data[siiys_group].get('num_sizes', 0),
                'sizes_available': [s['size_label'] for s in siiys_data[siiys_group].get('sizes', [])],
            }
        
        # Download product images (skip if already exists)
        raw_images = raw.get('images', [])
        for j, img in enumerate(raw_images[:MAX_IMG_PER_PRODUCT]):
            src = img.get('src', '')
            if not src:
                continue
            
            safe_handle = re.sub(r'[^a-zA-Z0-9_-]', '_', handle)[:50]
            img_hash = hashlib.md5(src.encode()).hexdigest()[:8]
            fname = f"{safe_handle}_{j+1}_{img_hash}.jpg"
            save_path = IMG_DIR / fname
            
            if save_path.exists() and save_path.stat().st_size > 1000:
                skipped_imgs += 1
                product['images'].append({
                    'url': src,
                    'alt': img.get('alt', ''),
                    'local': f"images/{fname}",
                    'width': img.get('width'),
                    'height': img.get('height'),
                })
            elif download_img(src, save_path):
                img_count += 1
                product['images'].append({
                    'url': src,
                    'alt': img.get('alt', ''),
                    'local': f"images/{fname}",
                    'width': img.get('width'),
                    'height': img.get('height'),
                })
        
        extracted.append(product)
        
        # Checkpoint every 100 products
        if (i + 1) % 100 == 0:
            save_checkpoint({
                'phase': 'products',
                'last_idx': i,
                'products': extracted,
                'img_count': img_count,
                'skipped': skipped_imgs,
            })
            print(f"    Checkpoint: {i+1} products, {img_count} new images, {skipped_imgs} skipped")
    
    print(f"\n  Product processing complete:")
    print(f"    Total products: {len(extracted)}")
    print(f"    New images downloaded: {img_count}")
    print(f"    Images already existed: {skipped_imgs}")
    
    return extracted


# ═══════════════════════════════════════════════════════════════════
# PHASE 5: SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════
def save_all(extracted, siiys_data):
    """Save final products.json and siiys_data.json."""
    print(f"\n{'='*60}")
    print("  PHASE 5: Saving structured data")
    print(f"{'='*60}")
    
    # Save products.json
    with open(PRODUCTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {PRODUCTS_FILE} ({len(extracted)} products)")
    
    # Save siiys_data.json
    with open(SIIYS_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(siiys_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {SIIYS_DATA_FILE} ({len(siiys_data)} groups)")
    
    # Stats
    siiys_products = [p for p in extracted if p.get('siiys_group')]
    with_images = [p for p in extracted if p['images']]
    with_model = [p for p in extracted if p['model_height']]
    with_fabric = [p for p in extracted if p['fabric']]
    with_fit = [p for p in extracted if p['fit_type']]
    with_sizes = [p for p in extracted if p['sizes']]
    
    print(f"\n  ═══ STATISTICS ═══")
    print(f"  Total products:         {len(extracted)}")
    print(f"  With images:            {len(with_images)}")
    print(f"  With sizes/variants:    {len(with_sizes)}")
    print(f"  With model info:        {len(with_model)}")
    print(f"  With fabric info:       {len(with_fabric)}")
    print(f"  With fit type:          {len(with_fit)}")
    print(f"  SIIYS products:         {len(siiys_products)}")
    print(f"  SIIYS groups:           {len(siiys_data)}")
    
    total_siiys_images = sum(
        len(g.get('sizes', []))
        for g in siiys_data.values()
    )
    print(f"  SIIYS size images:      {total_siiys_images}")
    
    total_measurements = sum(
        len(g.get('measurements', []))
        for g in siiys_data.values()
    )
    print(f"  SIIYS measurements:     {total_measurements}")
    
    # Product types breakdown
    types = {}
    for p in extracted:
        pt = p.get('product_type', 'Unknown') or 'Unknown'
        types[pt] = types.get(pt, 0) + 1
    
    print(f"\n  Product types:")
    for pt, count in sorted(types.items(), key=lambda x: -x[1])[:15]:
        print(f"    {pt:30s}: {count}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  UNIVERSAL STANDARD — Complete Extraction")
    print("  Focus: SIIYS paired multi-size images + measurements")
    print("=" * 60)
    
    # Create directories
    for d in [BASE_DIR, IMG_DIR, SIIYS_DIR, SIIYS_GROUPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Fetch catalog
    all_products = fetch_all_products()
    
    if not all_products:
        print("\nERROR: No products fetched. Check network connection.")
        exit(1)
    
    # Phase 2: Identify SIIYS groups
    groups = identify_siiys_groups(all_products)
    
    # Phase 3: Extract SIIYS images + measurements
    siiys_data = extract_all_siiys(groups)
    
    # Phase 4: Process all products + download images
    extracted = process_products(all_products, siiys_data)
    
    # Phase 5: Save everything
    save_all(extracted, siiys_data)
    
    # Cleanup checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    
    print(f"\n{'='*60}")
    print("  EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {BASE_DIR.absolute()}")
    print(f"  Products: {PRODUCTS_FILE.name}")
    print(f"  SIIYS data: {SIIYS_DATA_FILE.name}")
    print(f"  SIIYS groups: {SIIYS_GROUPS_DIR.name}/")
