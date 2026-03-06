"""
FitFusion — Data Preprocessing Pipeline
==========================================
Converts Snag Tights + Universal Standard SIIYS raw data into VITON-HD format
suitable for training with train_xl_qlora.py.

Two stages:
  Stage 1 (Local/RunPod): Organize images + create measurements.json + train_pairs.txt
  Stage 2 (RunPod only):  Generate DensePose maps + agnostic masks

Usage:
  python prepare_vitonhd_dataset.py --stage 1 --output_dir data/fitfusion_vitonhd
  python prepare_vitonhd_dataset.py --stage 2 --output_dir data/fitfusion_vitonhd
  python prepare_vitonhd_dataset.py --stage all --output_dir data/fitfusion_vitonhd

Output structure:
  data/fitfusion_vitonhd/
      train/
          image/              ← person photos
          cloth/              ← garment reference images
          agnostic-mask/      ← *_mask.png (stage 2)
          image-densepose/    ← DensePose maps (stage 2)
      test/
          image/ cloth/ agnostic-mask/ image-densepose/
      train_pairs.txt
      test_pairs.txt
      measurements.json
      vitonhd_train_tagged.json
      vitonhd_test_tagged.json
      dataset_stats.json
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict


# ─── Configuration ───────────────────────────────────────────────────────────

# Directories relative to project root
SNAG_DIR = "training_data_extraction/snag_tights"
US_DIR = "training_data_extraction/universal_standard"

# VITON-HD image size
TARGET_SIZE = (768, 1024)  # (width, height) — IDM-VTON default

# Train/test split
TEST_RATIO = 0.1

# Cross-size pairing: for each garment, generate pairs between all available sizes
CROSS_SIZE_PAIRS = True


# ─── Measurement Conversion Utilities ────────────────────────────────────────

def parse_cm_range(val: str) -> float:
    """Parse a cm range like '101-106' → midpoint 103.5."""
    if not val:
        return 0.0
    val = val.strip()
    if "-" in val:
        parts = val.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def parse_imperial_inches(val: str) -> float:
    """Parse inches like '33"' → cm (83.82). Also handles feet like 5' 9.5"."""
    if not val:
        return 0.0
    val = val.strip().replace('"', "").replace("'", "'")

    # Height format: 5' 9.5
    if "'" in val:
        parts = val.split("'")
        feet = float(parts[0].strip())
        inches = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return (feet * 12 + inches) * 2.54

    try:
        return float(val) * 2.54
    except ValueError:
        return 0.0


def snag_size_to_label(size_code: str) -> str:
    """Map Snag Tights size codes to universal labels."""
    # Snag uses: A/B, A, B, C, C1, C2, D, D1, D2, ... J, J1, J2
    # Map to approximate universal sizes
    mapping = {
        "A/B": "XS", "A": "XS", "B": "S",
        "C": "S", "C1": "S", "C2": "M",
        "D": "M", "D1": "M", "D2": "L",
        "E": "L", "E1": "L", "E2": "XL",
        "F": "XL", "F1": "XL", "F2": "2XL", "Short F": "XL",
        "G": "2XL", "G1": "2XL", "G2": "3XL",
        "H": "3XL", "H1": "3XL", "H2": "4XL",
        "I": "4XL", "I1": "4XL", "I2": "4XL",
        "J": "4XL", "J1": "4XL", "J2": "4XL",
    }
    return mapping.get(size_code, "M")


def garment_type_from_title(title: str) -> str:
    """Infer garment type from product title."""
    title_lower = title.lower()
    type_keywords = {
        "tights": "tights",
        "legging": "tights",
        "jogger": "pants",
        "jeans": "jeans",
        "pants": "pants",
        "trousers": "pants",
        "dress": "dress",
        "bodysuit": "bodysuit",
        "tee": "top",
        "t-shirt": "top",
        "shirt": "top",
        "top": "top",
        "jacket": "jacket",
        "coat": "jacket",
        "hoodie": "hoodie",
        "sweater": "hoodie",
        "skirt": "skirt",
        "shorts": "pants",
        "aviator": "jacket",
    }
    for keyword, gtype in type_keywords.items():
        if keyword in title_lower:
            return gtype
    return "top"  # Default


# ─── Stage 1: Organize Images + Create Metadata ─────────────────────────────

def stage1_organize(output_dir: str, project_root: str):
    """
    Stage 1: Copy images into VITON-HD structure and create metadata files.
    """
    print("=" * 70)
    print("Stage 1: Organizing images into VITON-HD format")
    print("=" * 70)

    out = Path(output_dir)
    for phase in ["train", "test"]:
        for subdir in ["image", "cloth", "agnostic-mask", "image-densepose"]:
            (out / phase / subdir).mkdir(parents=True, exist_ok=True)

    all_entries = []  # List of dicts with image paths + measurements + metadata

    # ── Process Universal Standard SIIYS ──
    us_entries = _process_universal_standard(project_root)
    all_entries.extend(us_entries)
    print(f"  Universal Standard: {len(us_entries)} entries")

    # ── Process Snag Tights ──
    snag_entries = _process_snag_tights(project_root)
    all_entries.extend(snag_entries)
    print(f"  Snag Tights:        {len(snag_entries)} entries")

    if not all_entries:
        print("ERROR: No data entries found. Check data directories.")
        return

    # ── Shuffle and split ──
    random.seed(42)
    random.shuffle(all_entries)
    n_test = max(1, int(len(all_entries) * TEST_RATIO))
    test_entries = all_entries[:n_test]
    train_entries = all_entries[n_test:]

    print(f"\n  Total entries: {len(all_entries)}")
    print(f"  Train: {len(train_entries)}, Test: {n_test}")

    # ── Copy images and build metadata ──
    measurements = {}
    train_pairs = []
    test_pairs = []
    train_tagged = {}
    test_tagged = {}
    img_counter = 0

    # Track mapping from entry → assigned image name (for cross-size pairing)
    entry_to_imgname = {}

    for phase, entries, pairs_list, tagged_dict in [
        ("train", train_entries, train_pairs, train_tagged),
        ("test", test_entries, test_pairs, test_tagged),
    ]:
        for entry_idx, entry in enumerate(entries):
            img_counter += 1
            img_name = f"{img_counter:06d}.jpg"
            cloth_name = img_name

            # Copy person image
            src_img = entry["person_image"]
            if os.path.exists(src_img):
                _copy_and_resize(src_img, out / phase / "image" / img_name)
            else:
                print(f"  [WARN] Missing person image: {src_img}")
                continue

            # Copy cloth image (or use same image if no separate cloth)
            src_cloth = entry.get("cloth_image", src_img)
            if os.path.exists(src_cloth):
                _isolate_cloth_and_resize(src_cloth, out / phase / "cloth" / cloth_name)
            else:
                # Fall back to person image
                _isolate_cloth_and_resize(src_img, out / phase / "cloth" / cloth_name)

            # Pairs
            pairs_list.append(f"{img_name} {cloth_name}")

            # Measurements
            measurements[img_name] = entry["measurements"]

            # Track entry → name mapping for cross-size pairing
            entry["_assigned_name"] = img_name
            entry["_assigned_phase"] = phase

            # Tagged annotations
            tagged_dict[img_name] = [
                {
                    "file_name": img_name,
                    "tag_info": [
                        {"tag_name": "item", "tag_category": entry.get("garment_type", "clothing")},
                        {"tag_name": "sleeveLength", "tag_category": entry.get("sleeve_length", None)},
                        {"tag_name": "neckLine", "tag_category": entry.get("neck_line", None)},
                    ],
                }
            ]

    # ── Generate cross-size training pairs (train only) ──
    if CROSS_SIZE_PAIRS:
        cross_pairs = _generate_cross_size_pairs(train_entries, train_pairs, measurements)
        train_pairs.extend(cross_pairs)
        print(f"  Cross-size pairs added: {len(cross_pairs)}")

    # ── Write files ──
    with open(out / "train_pairs.txt", "w") as f:
        f.write("\n".join(train_pairs) + "\n")
    with open(out / "test_pairs.txt", "w") as f:
        f.write("\n".join(test_pairs) + "\n")
    with open(out / "measurements.json", "w") as f:
        json.dump(measurements, f, indent=2)
    with open(out / "train" / "vitonhd_train_tagged.json", "w") as f:
        json.dump(train_tagged, f, indent=2)
    with open(out / "test" / "vitonhd_test_tagged.json", "w") as f:
        json.dump(test_tagged, f, indent=2)

    # Dataset stats
    stats = {
        "total_entries": len(all_entries),
        "train_entries": len(train_entries),
        "test_entries": len(test_entries),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "measurements_count": len(measurements),
        "sources": {
            "universal_standard_siiys": len(us_entries),
            "snag_tights": len(snag_entries),
        },
    }
    with open(out / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Stage 1 complete!")
    print(f"  Output: {output_dir}")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Test pairs:  {len(test_pairs)}")
    print(f"  Measurements: {len(measurements)}")
    print(f"\n  Next: Run Stage 2 on RunPod to generate DensePose + masks")


def _process_universal_standard(project_root: str) -> list:
    """Process Universal Standard SIIYS data."""
    entries = []
    siiys_path = os.path.join(project_root, US_DIR, "siiys_data.json")

    if not os.path.exists(siiys_path):
        print(f"  [WARN] Universal Standard data not found: {siiys_path}")
        return entries

    with open(siiys_path) as f:
        siiys_data = json.load(f)

    for group_name, group_data in siiys_data.items():
        garment_type = garment_type_from_title(group_name)

        # Collect all sizes with valid images
        valid_sizes = []
        for size_info in group_data.get("sizes", []):
            local_path = size_info.get("local_path", "")
            full_path = os.path.join(project_root, US_DIR, local_path)

            if os.path.exists(full_path):
                model_info = size_info.get("model", {})
                valid_sizes.append({
                    "path": full_path,
                    "size_label": size_info.get("size_label", "M"),
                    "bust_cm": parse_imperial_inches(model_info.get("bust", "")),
                    "waist_cm": parse_imperial_inches(model_info.get("waist", "")),
                    "hips_cm": parse_imperial_inches(model_info.get("hips", "")),
                    "height_cm": parse_imperial_inches(model_info.get("height", "")),
                    "group": group_name,
                })

        # For each size, create an entry
        # The "cloth" image can be from any other size
        for i, sz in enumerate(valid_sizes):
            # Use a different size's image as cloth reference (cross-size)
            # For self-reconstruction, cloth = same image
            cloth_idx = (i + 1) % len(valid_sizes) if len(valid_sizes) > 1 else i

            entry = {
                "person_image": sz["path"],
                "cloth_image": valid_sizes[cloth_idx]["path"],
                "measurements": {
                    "bust_cm": sz["bust_cm"],
                    "waist_cm": sz["waist_cm"],
                    "hips_cm": sz["hips_cm"],
                    "height_cm": sz["height_cm"],
                    "size_label": sz["size_label"],
                    "garment_type": garment_type,
                },
                "garment_type": garment_type,
                "source": "universal_standard",
                "group": sz["group"],
                "size_label": sz["size_label"],
            }
            entries.append(entry)

    return entries


def _process_snag_tights(project_root: str) -> list:
    """Process Snag Tights paired data."""
    entries = []
    paired_path = os.path.join(project_root, SNAG_DIR, "paired_data.json")
    size_chart_path = os.path.join(project_root, SNAG_DIR, "size_chart.json")

    if not os.path.exists(paired_path):
        print(f"  [WARN] Snag Tights data not found: {paired_path}")
        return entries

    with open(paired_path) as f:
        paired_data = json.load(f)

    size_chart = {}
    if os.path.exists(size_chart_path):
        with open(size_chart_path) as f:
            size_chart = json.load(f)

    images_dir = os.path.join(project_root, SNAG_DIR, "paired_images")

    for product in paired_data:
        handle = product.get("handle", "")
        title = product.get("title", "")
        garment_type = garment_type_from_title(title)
        sizes = product.get("sizes", {})

        # Collect all size images for this product
        product_sizes = []
        for size_code, size_data in sizes.items():
            meas = size_data.get("measurements", {})
            if not meas:
                meas = size_chart.get(size_code, {})

            # Scan actual files on disk instead of using filenames from JSON
            # (JSON has original Shopify names, disk has ModelName_shotN.jpg)
            size_dir = os.path.join(images_dir, handle, f"size_{size_code}")
            if not os.path.isdir(size_dir):
                continue

            actual_files = [
                f for f in os.listdir(size_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for filename in actual_files:
                img_path = os.path.join(size_dir, filename)
                product_sizes.append({
                    "path": img_path,
                    "size_code": size_code,
                    "size_label": snag_size_to_label(size_code),
                    "bust_cm": 0.0,  # Snag doesn't provide bust
                    "waist_cm": parse_cm_range(meas.get("waist_cm", "")),
                    "hips_cm": parse_cm_range(meas.get("hip_cm", "")),
                    "height_cm": parse_cm_range(meas.get("height_cm", "")),
                    "garment_type": garment_type,
                })

        # Create entries — use first image as cloth reference for all
        if product_sizes:
            cloth_img = product_sizes[0]["path"]
            for sz in product_sizes:
                entry = {
                    "person_image": sz["path"],
                    "cloth_image": cloth_img,
                    "measurements": {
                        "bust_cm": sz["bust_cm"],
                        "waist_cm": sz["waist_cm"],
                        "hips_cm": sz["hips_cm"],
                        "height_cm": sz["height_cm"],
                        "size_label": sz["size_label"],
                        "garment_type": sz["garment_type"],
                    },
                    "garment_type": sz["garment_type"],
                    "source": "snag_tights",
                    "product_handle": handle,
                    "size_code": sz["size_code"],
                }
                entries.append(entry)

    return entries


def _generate_cross_size_pairs(entries: list, existing_pairs: list, measurements: dict) -> list:
    """
    Generate cross-size training pairs: for same garment, pair each person image
    with cloth from a different size. This teaches the model size-awareness.
    
    For each group/product with multiple sizes, we create pairs like:
      person_image_size_A  cloth_image_size_B
    The measurement_encoder gets size_A's measurements, so the model learns
    "given these body measurements, fit this garment to this body shape."
    
    Returns list of "img_name cloth_name" strings to append to train_pairs.txt.
    """
    cross_pairs = []

    # Group entries by source+group (for US) or source+product (for Snag)
    groups = defaultdict(list)
    for entry in entries:
        # Only use entries that were successfully assigned (have _assigned_name)
        if "_assigned_name" not in entry:
            continue
        if entry.get("_assigned_phase") != "train":
            continue

        if entry["source"] == "universal_standard":
            key = f"us_{entry['group']}"
        else:
            key = f"snag_{entry['product_handle']}"
        groups[key].append(entry)

    pairs_added = 0
    for key, group_entries in groups.items():
        if len(group_entries) < 2:
            continue

        # For each entry, pair it with cloth from every OTHER size in the group
        for i, person_entry in enumerate(group_entries):
            person_img = person_entry["_assigned_name"]

            for j, cloth_entry in enumerate(group_entries):
                if i == j:
                    continue  # Skip self-pair (already exists)

                cloth_img = cloth_entry["_assigned_name"]
                pair_str = f"{person_img} {cloth_img}"

                # Avoid duplicate pairs
                if pair_str not in existing_pairs and pair_str not in cross_pairs:
                    cross_pairs.append(pair_str)
                    pairs_added += 1

    print(f"    Groups with 2+ sizes: {sum(1 for g in groups.values() if len(g) >= 2)}")
    print(f"    Cross-size pairs generated: {pairs_added}")

    return cross_pairs


def _copy_and_resize(src: str, dst: Path):
    """Copy and resize image to VITON-HD target size."""
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(str(dst), quality=95)
    except Exception as e:
        print(f"  [ERR] Failed to process {src}: {e}")


def _isolate_cloth_and_resize(src: str, dst: Path):
    """Isolate garment (remove background/person) and resize."""
    try:
        img = Image.open(src).convert("RGB")
        try:
            pass # from rembg import remove
            # img = remove(img).convert("RGB")
        except ImportError:
            pass  # Fallback to original
        
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(str(dst), quality=95)
    except Exception as e:
        print(f"  [ERR] Failed to process cloth {src}: {e}")


# ─── Stage 2: Generate DensePose + Agnostic Masks (RunPod) ──────────────────

def stage2_preprocessing(output_dir: str):
    """
    Stage 2: Generate DensePose maps and agnostic masks.
    Must run on Linux (RunPod) with Detectron2 installed.
    """
    print("=" * 70)
    print("Stage 2: Generating DensePose maps + agnostic masks")
    print("=" * 70)

    out = Path(output_dir)

    for phase in ["train", "test"]:
        image_dir = out / phase / "image"
        densepose_dir = out / phase / "image-densepose"
        mask_dir = out / phase / "agnostic-mask"

        if not image_dir.exists():
            print(f"  [SKIP] No {phase}/image directory found")
            continue

        images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        print(f"\n  Processing {len(images)} {phase} images...")

        # ── DensePose ──
        _generate_densepose(images, densepose_dir)

        # ── Agnostic Masks ──
        _generate_agnostic_masks(images, mask_dir)

    print(f"\n✓ Stage 2 complete!")


def _generate_densepose(images: list, output_dir: Path):
    """
    Generate DensePose maps using Detectron2.
    Falls back to simple body-region segmentation if Detectron2 is unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try importing Detectron2 DensePose
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from densepose.config import add_densepose_config
        from densepose import add_densepose_config as adc
        import numpy as np
        import cv2

        print("  Using Detectron2 DensePose...")
        _densepose_detectron2(images, output_dir)

    except ImportError:
        print("  [WARN] Detectron2 not available. Using fallback pose estimation.")
        print("         For better results, run Stage 2 on RunPod with Detectron2.")
        _densepose_fallback(images, output_dir)


def _densepose_detectron2(images: list, output_dir: Path):
    """Generate DensePose using Detectron2 (Linux/RunPod)."""
    import numpy as np
    import cv2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    # Setup DensePose config
    cfg = get_cfg()
    try:
        from densepose import add_densepose_config
        add_densepose_config(cfg)
    except ImportError:
        from detectron2.projects.DensePose import add_densepose_config
        add_densepose_config(cfg)

    cfg.merge_from_file(
        "detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
    )
    cfg.MODEL.WEIGHTS = (
        "https://dl.fbaipublicfiles.com/densepose/"
        "densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    )
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)

    from tqdm import tqdm

    for img_path in tqdm(images, desc="  DensePose"):
        out_path = output_dir / img_path.name
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        outputs = predictor(img)

        # Extract DensePose visualization
        if outputs["instances"].has("pred_densepose"):
            densepose = outputs["instances"].pred_densepose
            # Create IUV visualization
            iuv = _densepose_to_iuv(densepose, img.shape[:2])
            cv2.imwrite(str(out_path), iuv)
        else:
            # No detection — save blank
            blank = np.zeros_like(img)
            cv2.imwrite(str(out_path), blank)


def _densepose_to_iuv(densepose_results, img_shape):
    """Convert DensePose results to IUV image."""
    import numpy as np

    h, w = img_shape
    iuv = np.zeros((h, w, 3), dtype=np.uint8)

    if hasattr(densepose_results, "labels_uv_confidences"):
        # Chart-based output
        for i in range(len(densepose_results)):
            result = densepose_results[i]
            box = result.proposal_boxes.tensor[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            labels = result.labels.cpu().numpy()
            u = result.uv[0].cpu().numpy()
            v = result.uv[1].cpu().numpy()

            h_box, w_box = labels.shape
            for dy in range(h_box):
                for dx in range(w_box):
                    py = y1 + dy
                    px = x1 + dx
                    if 0 <= py < h and 0 <= px < w and labels[dy, dx] > 0:
                        iuv[py, px, 0] = labels[dy, dx]
                        iuv[py, px, 1] = int(u[dy, dx] * 255)
                        iuv[py, px, 2] = int(v[dy, dx] * 255)

    return iuv


def _densepose_fallback(images: list, output_dir: Path):
    """
    Fallback: create IUV-style body region maps using MediaPipe.
    
    DensePose IUV format: 3-channel image where
      Channel 0 (I) = body part index (0-24)
      Channel 1 (U) = U surface coordinate (0-255)
      Channel 2 (V) = V surface coordinate (0-255)
    
    This fallback approximates the body part segmentation using MediaPipe's
    pose landmarks to create filled body regions (not just skeleton lines).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mediapipe as mp
        import numpy as np
        from PIL import Image as PILImage, ImageDraw
        import cv2

        try:
            mp_pose = mp.solutions.pose
        except AttributeError:
            print("    [WARN] mediapipe.solutions missing. Creating blank DensePose maps.")
            _create_blank_images(images, output_dir)
            return
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        from tqdm import tqdm

        # DensePose body part color mapping (I channel values for each region)
        # These match the 24 DensePose body parts
        BODY_PARTS = {
            "head": 1,
            "torso_front": 2,
            "torso_back": 3,
            "right_upper_arm": 4,
            "left_upper_arm": 5,
            "right_lower_arm": 6,
            "left_lower_arm": 7,
            "right_hand": 8,
            "left_hand": 9,
            "right_upper_leg": 10,
            "left_upper_leg": 11,
            "right_lower_leg": 12,
            "left_lower_leg": 13,
            "right_foot": 14,
            "left_foot": 15,
        }

        def _get_landmark_point(landmarks, idx, w, h):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        def _draw_polygon(iuv, points, part_id, w, h):
            """Draw a filled polygon for a body part with IUV values."""
            pts = np.array(points, dtype=np.int32)
            # I channel: body part index
            cv2.fillPoly(iuv[:, :, 0], [pts], int(part_id * 10))  # Scale for visibility
            # U channel: normalized x position within region
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                x_range = max(x_max - x_min, 1)
                y_range = max(y_max - y_min, 1)
                iuv[ys, xs, 1] = ((xs - x_min) * 255 / x_range).astype(np.uint8)
                iuv[ys, xs, 2] = ((ys - y_min) * 255 / y_range).astype(np.uint8)

        def _expand_point(center, ref_point, factor=0.3):
            """Expand a limb segment into a polygon with width."""
            dx = ref_point[0] - center[0]
            dy = ref_point[1] - center[1]
            length = max(np.sqrt(dx**2 + dy**2), 1)
            # Perpendicular direction for width
            nx = -dy / length * length * factor
            ny = dx / length * length * factor
            return (int(center[0] + nx), int(center[1] + ny)), (int(center[0] - nx), int(center[1] - ny))

        def _limb_polygon(p1, p2, width_factor=0.15):
            """Create a polygon for a limb segment with given width."""
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = max(np.sqrt(dx**2 + dy**2), 1)
            nx = -dy / length * width_factor * length
            ny = dx / length * width_factor * length
            # Clamp width
            max_width = 40
            scale = min(max_width / max(abs(nx), abs(ny), 1), 1.0)
            nx *= scale
            ny *= scale
            return [
                (int(p1[0] + nx), int(p1[1] + ny)),
                (int(p2[0] + nx), int(p2[1] + ny)),
                (int(p2[0] - nx), int(p2[1] - ny)),
                (int(p1[0] - nx), int(p1[1] - ny)),
            ]

        for img_path in tqdm(images, desc="  DensePose (fallback IUV)"):
            out_path = output_dir / img_path.name
            if out_path.exists():
                continue

            img = PILImage.open(str(img_path)).convert("RGB")
            w, h = img.size
            result = pose.process(np.array(img))

            iuv = np.zeros((h, w, 3), dtype=np.uint8)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                pt = lambda idx: _get_landmark_point(lm, idx, w, h)

                # MediaPipe landmark indices:
                # 0=nose, 7=left_ear, 8=right_ear, 11=left_shoulder, 12=right_shoulder
                # 13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist
                # 23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee
                # 27=left_ankle, 28=right_ankle

                try:
                    # Head (circle around nose/ears)
                    nose = pt(0)
                    head_radius = max(abs(pt(7)[0] - pt(8)[0]), 30)
                    cv2.circle(iuv[:, :, 0], nose, head_radius, BODY_PARTS["head"] * 10, -1)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mask, nose, head_radius, 255, -1)
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0:
                        iuv[ys, xs, 1] = 128
                        iuv[ys, xs, 2] = 128

                    # Torso (quadrilateral: shoulders to hips)
                    l_shoulder = pt(11)
                    r_shoulder = pt(12)
                    l_hip = pt(23)
                    r_hip = pt(24)
                    torso_pts = [l_shoulder, r_shoulder, r_hip, l_hip]
                    _draw_polygon(iuv, torso_pts, BODY_PARTS["torso_front"], w, h)

                    # Left upper arm
                    poly = _limb_polygon(pt(11), pt(13), 0.2)
                    _draw_polygon(iuv, poly, BODY_PARTS["left_upper_arm"], w, h)

                    # Left lower arm
                    poly = _limb_polygon(pt(13), pt(15), 0.15)
                    _draw_polygon(iuv, poly, BODY_PARTS["left_lower_arm"], w, h)

                    # Right upper arm
                    poly = _limb_polygon(pt(12), pt(14), 0.2)
                    _draw_polygon(iuv, poly, BODY_PARTS["right_upper_arm"], w, h)

                    # Right lower arm
                    poly = _limb_polygon(pt(14), pt(16), 0.15)
                    _draw_polygon(iuv, poly, BODY_PARTS["right_lower_arm"], w, h)

                    # Left upper leg
                    poly = _limb_polygon(pt(23), pt(25), 0.25)
                    _draw_polygon(iuv, poly, BODY_PARTS["left_upper_leg"], w, h)

                    # Left lower leg
                    poly = _limb_polygon(pt(25), pt(27), 0.2)
                    _draw_polygon(iuv, poly, BODY_PARTS["left_lower_leg"], w, h)

                    # Right upper leg
                    poly = _limb_polygon(pt(24), pt(26), 0.25)
                    _draw_polygon(iuv, poly, BODY_PARTS["right_upper_leg"], w, h)

                    # Right lower leg
                    poly = _limb_polygon(pt(26), pt(28), 0.2)
                    _draw_polygon(iuv, poly, BODY_PARTS["right_lower_leg"], w, h)

                except (IndexError, ValueError):
                    pass  # Incomplete pose detection

            # Save as RGB image (IUV encoded)
            iuv_img = PILImage.fromarray(iuv)
            iuv_img.save(str(out_path))

        pose.close()

    except ImportError:
        print("  [WARN] MediaPipe not available. Creating blank DensePose maps.")
        _create_blank_images(images, output_dir)

def _create_blank_images(images: list, output_dir: Path):
    from PIL import Image as PILImage
    import numpy as np
    from tqdm import tqdm
    for img_path in tqdm(images, desc="  Blank DensePose"):
        out_path = output_dir / img_path.name
        if out_path.exists():
            continue
        try:
            img = PILImage.open(str(img_path))
            w, h = img.size
        except:
            w, h = 768, 1024
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        PILImage.fromarray(blank).save(str(out_path))


def _generate_agnostic_masks(images: list, output_dir: Path):
    """
    Generate agnostic masks (garment region = 1, rest = 0).
    Uses human parsing or simple body segmentation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try SCHP (Self-Correction Human Parsing) or similar
        _masks_with_parsing(images, output_dir)
    except ImportError:
        print("  [WARN] Human parsing not available. Using GrabCut fallback.")
        _masks_fallback(images, output_dir)


def _masks_with_parsing(images: list, output_dir: Path):
    """Generate masks using PyTorch-based human parsing."""
    try:
        import numpy as np
        import cv2
        from transformers import pipeline as hf_pipeline

        # Use a segmentation model from Hugging Face
        print("  Loading segmentation model...")
        seg_model = hf_pipeline(
            "image-segmentation",
            model="mattmdjaga/segformer_b2_clothes",
            device=0 if __import__("torch").cuda.is_available() else -1,
        )

        from tqdm import tqdm

        for img_path in tqdm(images, desc="  Masks"):
            mask_name = img_path.stem + "_mask.png"
            out_path = output_dir / mask_name
            if out_path.exists():
                continue

            img = Image.open(str(img_path)).convert("RGB")
            results = seg_model(img)

            # Create binary mask: garment regions = white, else = black
            w, h = img.size
            mask = np.zeros((h, w), dtype=np.uint8)

            # Garment labels to mask
            garment_labels = [
                "Upper-clothes", "Dress", "Coat", "Pants", "Skirt",
                "Jumpsuits", "Scarf", "Left-shoe", "Right-shoe",
            ]

            for result in results:
                label = result.get("label", "")
                if label in garment_labels:
                    seg_mask = np.array(result["mask"])
                    if seg_mask.shape[:2] != (h, w):
                        seg_mask = cv2.resize(seg_mask, (w, h))
                    mask = np.maximum(mask, (seg_mask > 128).astype(np.uint8) * 255)

            # Apply dilation to agnostic mask to give model "baggy room"
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            mask_img = Image.fromarray(mask).convert("L")
            mask_img.save(str(out_path))

    except Exception as e:
        print(f"  [WARN] Parsing failed: {e}")
        print("  Falling back to simple mask generation.")
        _masks_fallback(images, output_dir)


def _masks_fallback(images: list, output_dir: Path):
    """
    Simple fallback: create approximate masks using GrabCut or fixed regions.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        print("  [WARN] OpenCV not available. Creating default masks.")
        _create_default_masks(images, output_dir)
        return

    from tqdm import tqdm

    for img_path in tqdm(images, desc="  Masks (GrabCut)"):
        mask_name = img_path.stem + "_mask.png"
        out_path = output_dir / mask_name
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # GrabCut with center rectangle as initial foreground
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Assume person is roughly centered
        rect = (int(w * 0.15), int(h * 0.05), int(w * 0.7), int(h * 0.9))
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

        # Focus on upper body region for clothing (top 60%)
        upper_mask = np.zeros_like(mask2)
        upper_mask[: int(h * 0.65), :] = mask2[: int(h * 0.65), :]

        # Apply dilation to agnostic mask to give model "baggy room"
        kernel = np.ones((15, 15), np.uint8)
        upper_mask = cv2.dilate(upper_mask, kernel, iterations=1)

        cv2.imwrite(str(out_path), upper_mask)


def _create_blank_images(images: list, output_dir: Path):
    """Create blank placeholder images."""
    from tqdm import tqdm

    for img_path in tqdm(images, desc="  Blanks"):
        out_path = output_dir / img_path.name
        if out_path.exists():
            continue
        img = Image.open(str(img_path)).convert("RGB")
        blank = Image.new("RGB", img.size, (0, 0, 0))
        blank.save(str(out_path))


def _create_default_masks(images: list, output_dir: Path):
    """Create default rectangular masks."""
    from tqdm import tqdm

    for img_path in tqdm(images, desc="  Default masks"):
        mask_name = img_path.stem + "_mask.png"
        out_path = output_dir / mask_name
        if out_path.exists():
            continue
        img = Image.open(str(img_path)).convert("RGB")
        w, h = img.size
        # White rectangle in center-upper region
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.6)],
            fill=255,
        )
        mask.save(str(out_path))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FitFusion data preprocessing pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="1",
        choices=["1", "2", "all"],
        help="Which stage to run (1=organize, 2=densepose+masks, all=both)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fitfusion_vitonhd",
        help="Output directory for VITON-HD format dataset",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Project root directory (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Auto-detect project root
    if args.project_root is None:
        # Script is in IDM-VTON/ or project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == "IDM-VTON":
            args.project_root = os.path.dirname(script_dir)
        else:
            args.project_root = script_dir

    print(f"Project root: {args.project_root}")
    print(f"Output dir:   {args.output_dir}")
    print()

    # Convert output_dir to absolute if relative
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(args.project_root, args.output_dir)

    if args.stage in ("1", "all"):
        stage1_organize(args.output_dir, args.project_root)

    if args.stage in ("2", "all"):
        stage2_preprocessing(args.output_dir)


if __name__ == "__main__":
    main()
