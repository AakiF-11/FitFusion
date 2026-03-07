# FitFusion — Testing Phase Log

> **Status:** Active — RunPod GPU inference working end-to-end as of March 7, 2026  
> **Test Subject:** Snag Tights "Cropped Borg Aviator Jacket — Black"  
> **Test Models:** Snag Tights catalog models A–D (bust 76–101cm)  
> **GPU:** RunPod A-series pod — CUDA 12.8 / PyTorch 2.10.0  

---

## 1. What We Are Testing

The goal of the testing phase is to validate one specific hypothesis before scaling to production:

> **"Can a single diffusion model (IDM-VTON) produce visually distinct, physically accurate garment renders across multiple clothing sizes — without any additional fine-tuning — purely through physics-driven preprocessing?"**

In practice, the test runs look like this:

```bash
python run_tryon.py \
  --product_id cropped-borg-aviator-jacket-black \
  --brand_id snag_tights \
  --s3_image_url file:///workspace/FitFusion/data/test_inputs/person_model_C_Pixie.jpg \
  --bust 92 --waist 76 --hips 100 --height 167 \
  --all_sizes \
  --output /workspace/output_all_sizes.png \
  --steps 30 --seed 42
```

`--all_sizes` generates one image per available size (L, XL, 3XL, 4XL) so we can visually compare whether the system correctly renders the jacket as increasingly oversized across the size ladder.

---

## 2. The Full Pipeline — From Customer Click to Output Image

When the `--all_sizes` flag runs, it loops through each garment size and executes this full pipeline for each one:

```
Customer Photo + Measurements
          │
          ▼
[Step 1] Brand Catalog Lookup (brand_catalog.py)
   • Load garments.json + models.json from data/brand_catalog/catalog.json
   • Derive model_id from JSON key (not a field inside the object)
   • Standardize brand-specific size codes (D1→L, E1→XL, G2→3XL, H2→4XL)
   • Match customer measurements to closest catalog model via Height-Gated
     Euclidean distance on (bust, waist, hips)
          │
          ▼
[Step 2] Fit Classification (size_aware_vton.py → classify_fit())
   • Compare garment size to matched model size → compute size_gap integer
   • Classify into: RELAXED / LOOSE / OVERSIZED / SNUG / TIGHT etc.
   • Compute ALL physics parameters from that single gap value:
       - width_ratio:           how much wider the garment is than the body
       - warp_intensity:        how aggressively to TPS-warp the garment
       - mask_expansion_px:     how many pixels to dilate/erode the inpaint mask
       - inpainting_strength:   how much creative freedom to give the model
       - edge_softness:         gaussian blur kernel size for mask edges
       - fabric_tension:        0.0 = draped/loose, 1.0 = stretched/taut
          │
          ▼
[Step 3] Background Removal (fitfusion/utils/preprocessing.py)
   • BRIA RMBG-1.4 (BiRefNet-based) → strips background, leaves person on alpha
   • Falls back to rembg/u2net if RMBG-1.4 model not in /workspace/model_cache/
   • Composites foreground over studio gray (RGB 238, 238, 238)
   • Purpose: prevent IDM-VTON from hallucinating background objects into garment
          │
          ▼
[Step 4] Garment Preprocessing (size_aware_vton.py → prepare_size_aware_inputs())
   Layer 1 — Garment Resize:   width_ratio applied ONLY to mask; garment pixel
                                tensor stays 1:1 (prevents texture hallucination)
   Layer 3 — TPS Warp:         garment image warped by warp_intensity using
                                TPS (Thin Plate Spline) toward body contour
                                Fallback: perspective warp if opencv-contrib missing
   Layer 4 — Regional Mask:    DensePose body-part IDs used to create distinct
                                mask zones (torso vs arms vs hips) each expanded
                                independently by mask_expansion_px
   Layer 5 — Gradient Edges:   Gaussian blur on mask edges — soft for LOOSE,
                                sharp for TIGHT fits
   Layer 6 — Inpainting Str:   larger size gaps → more creative latitude
   Layer 7 — Garment-Type:     separate logic branches for tops / bottoms / dresses
          │
          ▼
[Step 5] DensePose Extraction (gradio_demo/apply_net.py + bundled detectron2)
   • Uses densepose_rcnn_R_50_FPN_s1x.yaml + model_final_162be9.pkl checkpoint
   • Produces IUV body-part map (24 regions, each pixel labelled with part ID)
   • Used in Layer 4 masking AND in Layer 6 TPS warp source point computation
          │
          ▼
[Step 6] IDM-VTON Diffusion Inference (SDXL-based)
   • UNet (unet_hacked_tryon.py) — main inpainting backbone
   • UNet Encoder (unet_hacked_garmnet.py) — garment feature extractor
   • VAE — encodes/decodes 768×1024 px latents
   • IP-Adapter — injects garment appearance into cross-attention layers
   • 30 DDPM steps, cfg=2.0, seed=42
   • Output: 768×1024 RGB PNG
```

---

## 3. Size Comparison Results (First Successful Run)

The first full run produced 4 outputs at ~3.2 it/s on the pod:

```
     L (  L): RELAXED      W=1.06x  Warp=0.5   Tension=0.35
    XL ( XL): LOOSE        W=1.12x  Warp=0.3   Tension=0.20
   3XL (3XL): OVERSIZED    W=1.18x  Warp=0.2   Tension=0.00
   4XL (4XL): OVERSIZED    W=1.18x  Warp=0.2   Tension=0.00
```

Output files: `tryon_L.png`, `tryon_XL.png`, `tryon_3XL.png`, `tryon_4XL.png` (~1.7 MB each)

The system successfully generated different physics parameters per size. Both 3XL and 4XL land in OVERSIZED because the catalog's largest model (Model D, bust=101cm) is only 1 size away from 4XL — meaning the gap saturates at the OVERSIZED tier for both. This is expected behaviour.

---

## 4. Every Error We Hit and How We Fixed It

The testing phase was an iterative debugging process. Below is the complete error log in order.

---

### Error 1 — `file://` URL not handled
**When:** First run attempt  
**Error:** `ValueError: Invalid s3_image_url format: Must start with http or s3://`  
**Cause:** The `download_customer_asset()` function in `run_tryon.py` only handled `http://` and `s3://` schemes. For local testing we pass a `file:///workspace/...` path.  
**Fix:** Added a `file://` branch that strips the scheme and returns the bare filesystem path.  
**Commit:** `1ace5de`

---

### Error 2 — Wrong `catalog_dir` resolved path
**When:** Second run  
**Error:** `Catalog empty — auto-onboarding from expanded_dataset... No measurements.json found`  
**Cause:** `--catalog_dir` defaulted to the relative string `"data/brand_catalog"`. When `run_tryon.py` runs from inside `IDM-VTON/`, that resolves to `IDM-VTON/data/brand_catalog` which doesn't exist.  
**Fix:** Changed the default to an absolute path derived from `Path(__file__).resolve().parent.parent`:
```python
_project_root = str(Path(__file__).resolve().parent.parent)
_default_catalog = os.path.join(_project_root, "data", "brand_catalog")
```
**Commit:** `89f0613`

---

### Error 3 — Silent `ImportError` hiding real crash cause
**When:** Same run as Error 2  
**Error:** `ERROR: IDM-VTON source code not found` — but no detail on WHY  
**Cause:** A bare `except ImportError: pass` was swallowing the actual module error.  
**Fix:** Exposed the real error message:
```python
except ImportError as e:
    print(f"ERROR: IDM-VTON source code import failed: {e}")
    sys.exit(1)
```
**Commit:** `89f0613`

---

### Error 4 — `JSONDecodeError: Unexpected UTF-8 BOM`
**When:** Third run  
**Error:** `json.decoder.JSONDecodeError: Unexpected UTF-8 BOM`  
**Cause:** `catalog.json` was edited and saved on Windows, which wrote a UTF-8 BOM marker (`\xef\xbb\xbf`) at the start of the file. Python's default `open()` doesn't strip it.  
**Fix:**
```python
open(catalog_file, encoding='utf-8-sig')  # utf-8-sig automatically strips BOM
```
**Commit:** `fe204e1`

---

### Error 5 — `ModelProfile.__init__() got an unexpected keyword argument 'brand'`
**When:** Fourth run  
**Error:** `TypeError: ModelProfile.__init__() got an unexpected keyword argument 'brand'`  
**Cause:** The `catalog.json` models object uses the JSON key as the model ID, with fields `brand`, `snag_label`, `size_label` etc. The `_load_catalog()` function was naively unpacking the entire dict as kwargs into `ModelProfile()`, but `ModelProfile` only accepts `model_id`, `name`, `bust_cm`, `waist_cm`, `hips_cm`, `height_cm`, `size_label`.  
**Fix:** Rewrote model loading to extract only known fields, deriving `model_id` and `name` from the JSON key:
```python
self.models[k] = ModelProfile(
    model_id=k, name=m.get("name", k),
    bust_cm=m["bust_cm"], waist_cm=m["waist_cm"],
    hips_cm=m["hips_cm"], height_cm=m["height_cm"],
    size_label=m.get("size_label", "M"),
)
```
**Commit:** `a6f3cc7`

---

### Error 6 — `available_sizes()` returning empty list
**When:** Same commit investigation  
**Cause:** `available_sizes()` only returned sizes that had reference photos (`data.photos` non-empty). The Snag Tights catalog has no per-size photos — only a single `test_image` flat lay. So the method returned `[]`, meaning no sizes were passed to the generator.  
**Fix:** Return ALL sizes if none have photos:
```python
def available_sizes(self) -> List[str]:
    with_photos = [s for s, data in self.sizes.items() if data.photos]
    return with_photos if with_photos else list(self.sizes.keys())
```
**Commit:** `a6f3cc7`

---

### Error 7 — Garment sizes loaded under wrong keys
**When:** Same commit  
**Cause:** Snag Tights uses brand-specific size codes (`D1`, `E1`, `G2`, `H2`) as the JSON keys inside `sizes`. The `standard_equiv` field holds the actual standard labels (`L`, `XL`, `3XL`, `4XL`). The old loader used the raw brand code as the key, so size lookups like `get_reference_photo(..., "XL")` found nothing.  
**Fix:** Use `standard_equiv` as the dict key when loading `GarmentSize`:
```python
std_label = sd.get("standard_equiv", s)
sizes[std_label] = GarmentSize(size_label=std_label, ...)
```
**Commit:** `a6f3cc7`

---

### Error 8 — No garment image fallback
**When:** Same commit  
**Cause:** `get_reference_photo()` returned `None` when no per-size photos existed, causing a NoneType crash downstream. The `test_image` field on the garment was never consulted.  
**Fix:** Added `garment_image_path: str = ""` to `GarmentProfile`, populated from `test_image`, and used it as final fallback in `get_reference_photo()`:
```python
if not photos and garment.garment_image_path:
    return {"image_path": garment.garment_image_path, ...}
```
**Commit:** `a6f3cc7`

---

### Error 9 — `No module named 'einops'`
**When:** Fifth run  
**Error:** `ERROR: IDM-VTON source code import failed: No module named 'einops'`  
**Cause:** `einops` is listed in `runpod_setup.sh` but the pod was created before that setup was run properly. The pod environment was missing it.  
**Fix:** `pip install einops xformers` directly on the pod. `einops` was already in `runpod_setup.sh` — no code change needed.  
**Commit:** pod-only fix

---

### Error 10 — `cv2.createThinPlateSplineShapeTransformer` missing
**When:** Sixth run — first run to actually load the model and start generating  
**Error:** `AttributeError: module 'cv2' has no attribute 'createThinPlateSplineShapeTransformer'`  
**Cause:** TPS (Thin Plate Spline) warping is an `opencv-contrib` feature. The pod had `opencv-python-headless` (base opencv) which does not include contrib modules.  
**Fix (code):** Wrapped TPS in try/except, falling back to perspective warp (4-point homography) which is available in base opencv:
```python
try:
    tps = cv2.createThinPlateSplineShapeTransformer()
    ...
except AttributeError:
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    result = cv2.warpPerspective(garment_np, M, (w, h))
```
**Fix (setup):** Changed `runpod_setup.sh` to install `opencv-contrib-python-headless` everywhere.  
**Commit:** `5c8808b`

---

### Error 11 — `torchvision::nms does not exist`
**When:** Seventh run  
**Error:** `RuntimeError: operator torchvision::nms does not exist`  
**Cause:** A critical binary mismatch. The pod had torch `2.10.0+cu128` (CUDA 12.8) but torchvision `0.17.0` which was compiled against an older CUDA. The NMS operator wasn't registering because the ABI was incompatible.  
**Fix:** Reinstalled torchvision with the matching CUDA 12.8 wheel:
```bash
pip install torchvision --index-url https://download.pytorch.org/whl/cu128
# Result: torchvision-0.25.0+cu128
```
Updated `runpod_setup.sh` to auto-detect CUDA version at setup time:
```bash
NVCC_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+')
TORCH_IDX="cu128"  # or cu118 for older pods
pip install torchvision --index-url "https://download.pytorch.org/whl/${TORCH_IDX}"
```
**Commit:** `1acfb39` (setup fix)

---

### Error 12 — `No module named 'pycocotools'`
**When:** Eighth run — new error past the torch fix  
**Error:** `ModuleNotFoundError: No module named 'pycocotools'`  
**Cause:** The bundled `detectron2` inside `gradio_demo/` requires `pycocotools` to load. It's in `runpod_setup.sh` but wasn't installed on this pod.  
**Fix:** `pip install pycocotools portalocker iopath shapely` on pod. Added all four to `runpod_setup.sh`.  
**Commit:** `1acfb39`

---

### Error 13 — `No module named 'av'` (PyAV)
**When:** Ninth run  
**Error:** `ModuleNotFoundError: No module named 'av'`  
**Cause:** The bundled `densepose/__init__.py` eagerly imports `densepose.data.video.video_keyframe_dataset` which imports `av` (the PyAV video library). This code path is only needed for video datasets — never for single-image inference.  
**Fix (dual approach):**  
1. `pip install av` on pod (stops the crash immediately)  
2. Wrapped the optional dataset/evaluation imports in try/except in `gradio_demo/densepose/__init__.py` so they don't crash on import when missing:
```python
try:
    from .data.datasets import builtin
except (ImportError, Exception):
    pass
try:
    from .evaluation import DensePoseCOCOEvaluator
except (ImportError, Exception):
    pass
```
**Commit:** `36c658b`

---

### Error 14 — Background not being applied / Face looks cartoony
**When:** After first successful image generation (reviewing tryon_L.png etc.)  
**Observed:** The person photo (outdoor autumn scene, outdoor lighting) was fed directly into IDM-VTON without background removal. The diffusion model hallucinated the face and rendered the person in an artificial/cartoony style because the model was trained on clean studio photos. The background removal code exists in `fitfusion/utils/preprocessing.py` (`standardize_background()`) and is called in `tryon_api.py`, but the direct `run_tryon.py` pipeline bypasses `tryon_api` and calls `_do_inference()` directly — which opens the person photo without any background standardization.  
**Status:** Known issue — fix pending (see Section 6).

---

## 5. The 7 Physics Layers — What Each Does

`size_aware_vton.py` contains the entire physics engine. Here is what each layer is responsible for:

| Layer | Name | What it does | Status |
|-------|------|--------------|--------|
| 1 | Mathematical Sizing | Computes `width_ratio` and `length_ratio` from garment vs body measurements | ✅ Working |
| 2 | Physics Prompts | Builds text prompts for the diffusion model (currently stripped to neutral — fit is driven by mask geometry, not text) | ✅ Working |
| 3 | Garment Resize | Applies `width_ratio` ONLY to the mask boundary, not the garment image pixels — preserves texture/logos | ✅ Working |
| 4 | Regional Masking | Uses DensePose IUV body-part IDs to create per-zone masks (torso, arms, legs expanded independently) | ✅ Working |
| 5 | Gradient Edges | Applies Gaussian blur to mask edges — LOOSE fits get soft edges (fabric drape), TIGHT fits get hard edges | ✅ Working |
| 6 | TPS Warp | Thin Plate Spline warps the garment toward body contour with `warp_intensity` — falls back to perspective warp | ✅ Working (with fallback) |
| 7 | Garment-Type Rules | Separate code branches for tops vs bottoms vs dresses, hemline z-layering for tucked garments | ✅ Working |

---

## 6. Known Issues and What We Need to Fix

### Issue A — Background Removal Not Applied in `run_tryon.py`

**Priority: HIGH**

Background removal is fully implemented in `fitfusion/utils/preprocessing.py` via `standardize_background()`. It is correctly called in `tryon_api.py → generate_tryon()`. However, the direct inference entry point `run_tryon.py → _do_inference()` opens the person photo raw without calling `standardize_background()`. The outdoor autumn photo we tested with (Pixie, standing in a leaf-covered park) went directly into IDM-VTON, causing:
- Face reconstructed in "studio model" style (hallucination)
- Background scenery bleeding into the garment render
- Overall cartoonish/artificial appearance

**Proposed Fix:** Call `standardize_background()` inside `_do_inference()` before opening the person image, same as `tryon_api.py` already does.

**Additional note:** For the background removal to work at full quality, the BRIA RMBG-1.4 model (`briaai/RMBG-1.4`) must be downloaded to `/workspace/model_cache/RMBG-1.4` during pod setup. If missing, the system falls back to the older `rembg/u2net` model. Both are in `runpod_setup.sh` but neither has been verified as available on the current pod.

---

### Issue B — Universal Standard Models Not Used for Testing

**Priority: MEDIUM**

The current test uses Snag Tights catalog models (Model A/B/C/D). Universal Standard models have better photos: mature poses, clean studio backgrounds, diverse body representation across sizes (XS–5X). These are more representative of the real-world use case. Universal Standard garment/model data exists in `training_data_extraction/universal_standard/` but has not yet been added to the active `data/brand_catalog/catalog.json`. 

---

### Issue C — 3XL and 4XL Produce Identical Physics Parameters

**Priority: LOW**

Both 3XL and 4XL classify as OVERSIZED (`W=1.18x`, `Warp=0.2`, `Tension=0.00`) because the model matched is "Model C" (bust=92cm, size L) and the gap from L to 3XL and L to 4XL both exceed the OVERSIZED threshold. The `size_gap` integer saturates at 3 and both sizes get the same treatment. The output images will look identical.

This is mathematically correct behaviour — one body in two different "significantly oversized" garments should look the same. However for the toggle comparison to be meaningful, we probably need to add a finer-grained cap higher than OVERSIZED (e.g. VERY_OVERSIZED) with a slightly larger `width_ratio` step.

---

### Issue D — `accelerate` Not Found Warning

**Priority: LOW / Informational**

Every run prints:
```
Cannot initialize model with low cpu memory usage because `accelerate` was not found
```
This is a diffusers library warning. The model loads and runs fine without it. `accelerate` would improve RAM efficiency during model loading (host CPU → GPU transfer). Adding `pip install accelerate` to the pod would eliminate the warning and speed up model load slightly.

---

## 7. Current File Map

```
FitFusion/
├── IDM-VTON/
│   ├── run_tryon.py          Entry point — CLI + all-sizes loop
│   ├── tryon_api.py          API layer (for brand button integration)
│   ├── brand_catalog.py      Catalog loader — B2B garment/model data
│   ├── size_aware_vton.py    7-layer physics engine
│   ├── size_charts.py        Size ratio lookup tables
│   ├── tryon_pipeline.py     SDXL inpainting pipeline (IDM-VTON)
│   ├── unet_hacked_tryon.py  Modified UNet for dual-encoder try-on
│   ├── unet_hacked_garmnet.py  Garment feature extractor UNet
│   ├── gradio_demo/
│   │   ├── apply_net.py      DensePose extraction CLI
│   │   ├── detectron2/       Bundled detectron2 (pre-compiled)
│   │   └── densepose/        Bundled DensePose inference code
│   └── ckpt/
│       ├── densepose/        model_final_162be9.pkl
│       ├── humanparsing/     parsing_atr.onnx, parsing_lip.onnx
│       ├── openpose/         body_pose_model.pth
│       └── image_encoder/    CLIP image encoder
├── fitfusion/
│   ├── utils/
│   │   └── preprocessing.py  Background removal (RMBG-1.4 + rembg fallback)
│   └── masking/
│       ├── compositor.py     Skin restoration after inpainting
│       └── confidence_scorer.py  Mask quality gating
├── data/
│   ├── brand_catalog/
│   │   └── catalog.json      Active garment + model registry
│   ├── test_inputs/
│   │   └── person_model_C_Pixie.jpg  Test person photo (outdoor)
│   └── measurements/
│       ├── garments.json     Garment dimension database
│       └── models.json       Model measurement database
└── runpod_setup.sh           Pod environment setup (all pip installs)
```

---

## 8. Reproduction Steps — Run the Test Yourself

### Prerequisites on a fresh RunPod pod:
```bash
cd /workspace
git clone https://github.com/AakiF-11/FitFusion.git
cd FitFusion
bash runpod_setup.sh
```

### Manual package installs needed until next `runpod_setup.sh` re-run:
```bash
# These were installed manually on the current pod session
# They are now in runpod_setup.sh for fresh pods
pip install einops xformers fvcore pycocotools portalocker iopath shapely av timm ftfy
pip install torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-contrib-python-headless
```

### Run all-sizes inference:
```bash
cd /workspace/FitFusion/IDM-VTON
python run_tryon.py \
  --product_id cropped-borg-aviator-jacket-black \
  --brand_id snag_tights \
  --s3_image_url file:///workspace/FitFusion/data/test_inputs/person_model_C_Pixie.jpg \
  --bust 92 --waist 76 --hips 100 --height 167 \
  --all_sizes \
  --output /workspace/output_all_sizes.png \
  --steps 30 --seed 42
```

Expected output log (abridged):
```
Loading IDM-VTON model...
  [1/5] VAE...
  [2/5] UNet...
  [3/5] Text encoders...
  [4/5] Image encoder...
  [5/5] UNet encoder...
  ✓ Model loaded!

Generating 4 sizes: ['L', 'XL', '3XL', '4XL']

============================================================
  FitFusion — Size-Aware Try-On
  Product: cropped-borg-aviator-jacket-black | Size: L
  Fit type: RELAXED — Width: 1.06x
  Running IDM-VTON inference (30 steps)...
    Extracting DensePose structure...Done.
  100%|██████████| 30/30 [00:09<00:00,  3.23it/s]
  ✓ Try-on image saved: /workspace/tryon_L.png
...
  ✓ Try-on image saved: /workspace/tryon_4XL.png

Size Comparison Summary:
     L: RELAXED   W=1.06x  Warp=0.5  Tension=0.35
    XL: LOOSE     W=1.12x  Warp=0.3  Tension=0.20
   3XL: OVERSIZED W=1.18x  Warp=0.2  Tension=0.00
   4XL: OVERSIZED W=1.18x  Warp=0.2  Tension=0.00
```

---

## 9. Commits Made During Testing Phase

| Commit | Description |
|--------|-------------|
| `1ace5de` | Fix `file://` URL handling in `download_customer_asset()` |
| `89f0613` | Fix catalog_dir default path; expose hidden ImportError |
| `fe204e1` | Fix UTF-8 BOM in catalog.json read (`utf-8-sig`) |
| `a6f3cc7` | Comprehensive brand_catalog.py rewrite: field mapping, available_sizes(), garment image fallback |
| `5c8808b` | TPS warp fallback for opencv without contrib; use opencv-contrib in setup |
| `36c658b` | Wrap optional densepose imports in try/except (av, scipy, evaluation) |
| `1acfb39` | CUDA-adaptive torchvision install; add portalocker/iopath/shapely to setup |
