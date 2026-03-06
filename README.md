# FitFusion — Size-Aware Virtual Try-On

**Production-grade Size-Conditioned Virtual Try-On pipeline built on IDM-VTON (SDXL)**

> GitHub: `https://github.com/AakiF-11/FitFusion` | Branch: `master`  
> Inference Platform: RunPod (NVIDIA RTX A6000, 48 GB VRAM)  
> Base Model: IDM-VTON (Stable Diffusion XL fine-tuned for try-on)

---

## Project Overview

FitFusion wraps IDM-VTON in a 6-stage physics-informed pipeline. Instead of a generic one-size image overlay, it uses real garment measurements, size charts, and fabric physics to show exactly how a specific size will drape on a specific body — adjusting mask geometry, warp tension, and inpainting guidance accordingly.

The pipeline handles everything from raw input images to a cleaned, skin-composited final result with zero hardcoded values.

---

## Repository Structure

```
FitFusion/
│
├── run_pipeline.py                  # MAIN ENTRY POINT — 6-stage CLI pipeline
├── runpod_controller.py             # Pod lifecycle management (start/stop/sync)
├── runpod_setup.sh                  # RunPod dependency installer
├── start_training.sh                # QLoRA training launcher
├── extract_snag_tights.py           # Brand data extraction (Snag Tights)
├── extract_universal_standard.py    # Brand data extraction (Universal Standard)
├── requirements.txt                 # Python dependencies
│
├── src/                             # Modular pipeline implementation
│   ├── preprocessing/
│   │   ├── background.py            # rembg background stripping → studio gray
│   │   └── validator.py             # Image dimension/format gating
│   ├── pose/
│   │   ├── openpose.py              # OpenPose keypoint extraction
│   │   └── densepose.py             # DensePose UV map generation
│   ├── masking/
│   │   ├── human_parsing.py         # SCHP dual-ONNX segmentation runner
│   │   ├── confidence.py            # Arm-geometry confidence scorer
│   │   ├── adaptive_mask.py         # Size-adaptive agnostic mask scaler
│   │   └── compositor.py            # Skin/tattoo restoration compositor
│   ├── size_physics/
│   │   ├── charts.py                # Brand size chart lookup tables
│   │   ├── physics.py               # 7-layer fit physics engine
│   │   └── resizer.py               # Garment image resizer (width/length ratios)
│   ├── inference/
│   │   ├── model_loader.py          # IDM-VTON UNet/VAE/encoder loader
│   │   └── tryon.py                 # Diffusion inference executor
│   └── catalog/
│       └── brand.py                 # B2B brand catalog ingestion
│
├── IDM-VTON/                        # Upstream IDM-VTON repository (unmodified)
│   ├── inference.py                 # Original IDM-VTON inference script
│   ├── infer_size_aware.py          # Size-aware inference adapter
│   ├── run_tryon.py                 # Legacy end-to-end runner
│   ├── tryon_api.py                 # Celery/Redis API endpoint
│   ├── size_aware_vton.py           # Physics engine (source)
│   ├── size_aware_pipeline.py       # Garment resizer (source)
│   ├── size_charts.py               # Size chart data (source)
│   ├── size_adaptive_mask.py        # Adaptive mask logic (source)
│   ├── brand_catalog.py             # Brand catalog logic (source)
│   ├── measurement_encoder.py       # Body measurement encoder
│   ├── tps_warper.py                # Thin-plate spline garment warper
│   ├── train_xl.py                  # IDM-VTON full fine-tuning script
│   ├── train_xl_qlora.py            # IDM-VTON QLoRA fine-tuning script
│   ├── generate_densepose.py        # DensePose generation utility
│   ├── prepare_vitonhd_dataset.py   # VITON-HD dataset preparation
│   ├── size_evaluation.py           # Fit quality evaluation
│   ├── ckpt/                        # Model weights (gitignored)
│   │   ├── unet/
│   │   ├── unet_encoder/
│   │   ├── vae/
│   │   ├── text_encoder/
│   │   ├── text_encoder_2/
│   │   ├── tokenizer/
│   │   ├── tokenizer_2/
│   │   ├── image_encoder/
│   │   ├── scheduler/
│   │   ├── densepose/
│   │   ├── humanparsing/
│   │   └── openpose/
│   └── gradio_demo/                 # Gradio web UI
│
├── fitfusion/                       # Legacy application-level modules
│   ├── utils/preprocessing.py       # Background standardization (rembg)
│   └── masking/
│       ├── compositor.py            # Skin compositing logic
│       └── confidence_scorer.py     # Mask validity scoring
│
├── data/
│   ├── brand_catalog/               # Brand JSON manifests + product images
│   │   ├── catalog.json
│   │   ├── snag_tights/
│   │   └── universal_standard/
│   ├── fitfusion_vitonhd/           # VITON-HD formatted dataset
│   │   ├── train_pairs.txt
│   │   ├── test_pairs.txt
│   │   ├── measurements.json
│   │   ├── train/
│   │   └── test/
│   └── tryon_output/                # Historical inference outputs
│
├── training_data_extraction/        # Raw scraped brand data
│   ├── snag_tights/
│   └── universal_standard/
│
├── tests/
│   └── evaluation_matrix.py         # Fit quality evaluation matrix
│
├── MODEL_SELECTION_AND_TRAINING_PLAN.md   # IDM-VTON vs alternatives analysis
├── FitFusion_README.md                    # Detailed architecture narrative
└── .gitignore
```

---

## Pipeline Architecture

The pipeline is driven by `run_pipeline.py` and executes 6 sequential stages:

```
Customer Photo + Garment Photo
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 · Preprocessing                                    │
│  · rembg background removal → solid RGB(238,238,238) gray   │
│  · Image dimension/format validation (min 512×384)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 · Pose Estimation                                  │
│  · OpenPose → 18-joint body skeleton keypoints              │
│  · DensePose → UV surface map for guided inpainting         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 · Agnostic Mask Generation                         │
│  · SCHP human parsing segmentation                          │
│  · Arm-geometry confidence scoring (threshold 0.85)         │
│  · Size-adaptive mask scaling (target_size vs person_size)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 · Size Physics                                     │
│  · Fit class computation (7 classes: BURSTING → TENT)       │
│  · Width/length ratio derivation from brand size charts     │
│  · Garment image geometric resize (mask-only, not texture)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 · IDM-VTON Diffusion Inference                     │
│  · Load UNet + UNet_Encoder + VAE + CLIP encoders           │
│  · IP-Adapter cross-attention garment conditioning          │
│  · SDXL inpainting at 1024×768                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6 · Post-Processing                                  │
│  · cv2.addWeighted skin/tattoo compositing                  │
│  · Result PNG + metadata.json saved to timestamped job dir  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Basic inference
python run_pipeline.py \
    --person_image  inputs/person/customer.jpg \
    --garment_image inputs/garment/jacket.jpg  \
    --target_size   XL                         \
    --ckpt_dir      ./ckpt                     \
    --output_dir    ./outputs

# Full options
python run_pipeline.py \
    --person_image  inputs/person/customer.jpg \
    --garment_image inputs/garment/jacket.jpg  \
    --target_size   XL                         \
    --person_size   M                          \
    --garment_type  jacket                     \
    --ckpt_dir      ./ckpt                     \
    --output_dir    ./outputs                  \
    --steps         30                         \
    --guidance_scale 2.0                       \
    --seed          42                         \
    --device        cuda                       \
    --save_intermediates
```

### CLI Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--person_image` | required | Full-body customer photo (JPG/PNG) |
| `--garment_image` | required | Flat garment product photo |
| `--output_dir` | `./outputs` | Result directory |
| `--target_size` | required | Size to simulate: `4XS`–`4XL` |
| `--person_size` | `M` | Customer's body baseline size |
| `--garment_type` | `top` | `top`, `pants`, `dress`, `skirt`, `jacket`, `outerwear` |
| `--ckpt_dir` | `./ckpt` | IDM-VTON weights directory |
| `--steps` | `30` | Diffusion inference steps |
| `--guidance_scale` | `2.0` | Classifier-free guidance scale |
| `--width` / `--height` | `768` / `1024` | Inference resolution |
| `--seed` | `42` | Reproducibility seed |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--skip_preprocessing` | off | Skip background stripping |
| `--skip_skin_restore` | off | Skip skin compositing |
| `--save_intermediates` | off | Save masked/warped/densepose images |

Output is written to `outputs/<person_stem>_<YYYYMMDD_HHMMSS>/result.png` + `metadata.json`.

---

## Checkpoint Layout

The `--ckpt_dir` must contain these subdirectories (all weights are gitignored):

```
ckpt/
├── unet/              # IDM-VTON fine-tuned UNet (SDXL)
├── unet_encoder/      # Garment feature encoder UNet
├── vae/               # Stable Diffusion XL VAE
├── text_encoder/      # CLIP ViT-L/14
├── text_encoder_2/    # CLIP ViT-bigG/14
├── tokenizer/         # CLIP tokenizer
├── tokenizer_2/       # CLIP tokenizer 2
├── image_encoder/     # IP-Adapter image encoder
├── scheduler/         # DDPM noise scheduler
├── densepose/         # Detectron2 DensePose weights
├── humanparsing/      # SCHP (parsing_atr.onnx, parsing_lip.onnx)
└── openpose/          # OpenPose body_pose_model.pth
```

On RunPod the weights live at `/workspace/FitFusion/ckpt/`.

---

## Fit Classes

The physics engine maps (person_size, target_size) pairs to one of 7 fit classes, each carrying distinct warp tension parameters:

| Class | Ratio H | Description |
|---|---|---|
| `BURSTING` | H < 0.85 | Extreme tension — garment far too small |
| `TIGHT` | 0.85 – 0.92 | Form-fitting |
| `SNUG` | 0.92 – 0.98 | Close fit |
| `PERFECT` | 0.98 – 1.08 | Classic/intended fit |
| `RELAXED` | 1.08 – 1.20 | Loose / oversized |
| `BAGGY` | 1.20 – 1.40 | Oversized |
| `TENT` | H > 1.40 | Extremely oversized |

*H = garment_measurement / body_measurement*

---

## `src/` Module Status

| Module | File | Status |
|---|---|---|
| Preprocessing | `src/preprocessing/background.py` | Complete |
| Preprocessing | `src/preprocessing/validator.py` | Complete |
| Pose | `src/pose/openpose.py` | Complete — wraps `IDM-VTON/preprocess/openpose/` (`OpenposeDetector`, ckpt monkey-patch) |
| Pose | `src/pose/densepose.py` | Complete — wraps `IDM-VTON/generate_densepose.py` (`segformer_b2_clothes` IUV approx) |
| Masking | `src/masking/human_parsing.py` | Complete — wraps `IDM-VTON/preprocess/humanparsing/` (dual ONNX: ATR + LIP) |
| Masking | `src/masking/confidence.py` | Complete (re-exports `fitfusion/masking/confidence_scorer.py`) |
| Masking | `src/masking/adaptive_mask.py` | Complete (wraps `IDM-VTON/size_adaptive_mask.py`) |
| Masking | `src/masking/compositor.py` | Complete (re-exports `fitfusion/masking/compositor.py`) |
| Size Physics | `src/size_physics/charts.py` | Complete (re-exports `IDM-VTON/size_charts.py`) |
| Size Physics | `src/size_physics/physics.py` | Complete (re-exports `IDM-VTON/size_aware_vton.py`) |
| Size Physics | `src/size_physics/resizer.py` | Complete (re-exports `IDM-VTON/size_aware_pipeline.py`) |
| Inference | `src/inference/model_loader.py` | Complete |
| Inference | `src/inference/tryon.py` | Complete |
| Catalog | `src/catalog/brand.py` | Complete (re-exports `IDM-VTON/brand_catalog.py`) |

**All modules are now implemented.** Stages 2 and 3 of `run_pipeline.py` are fully wired with no `NotImplementedError` stubs remaining.

### Implementation Details

**`src/pose/openpose.py`** — Lazy-loads `OpenposeDetector` from `IDM-VTON/preprocess/openpose/annotator/openpose/`. Monkey-patches `annotator_ckpts_path` on both `annotator.util` and `annotator.openpose` modules before instantiation, so `body_pose_model.pth` is resolved from the caller-supplied `ckpt_dir` regardless of the symlink state. Returns `{pose_keypoints_2d: list[18], pose_image: PIL}` — both needed by downstream stages.

**`src/pose/densepose.py`** — Uses the segformer-based IUV approximation from `IDM-VTON/generate_densepose.py`. Detectron2 DensePose is excluded because it cannot be installed cleanly on CUDA 12.1. Loads `mattmdjaga/segformer_b2_clothes` from local HF snapshot (offline-first) or HF Hub. Returns a PIL IUV RGB image matching the input resolution.

**`src/masking/human_parsing.py`** — Creates `onnxruntime.InferenceSession` objects directly from `ckpt_dir`, bypassing `run_parsing.py`'s hardcoded paths. Passes PIL Image directly to `onnx_inference()` — `SimpleFolderDataset` handles `isinstance(root, Image.Image)` natively, meaning zero filesystem I/O. ATR session uses CUDA; LIP (neck refinement) runs on CPU to avoid VRAM contention. Returns `(H, W)` uint8 numpy array of SCHP class labels 0–18.

---

## RunPod Deployment

**Pod:** `hn05v8n20u7btj-6441183f@ssh.runpod.io`  
**GPU:** NVIDIA RTX A6000 (48 GB VRAM)  
**SSH key:** `C:\Users\Aakif\.ssh\id_ed25519`

```bash
# Connect
ssh -i C:\Users\Aakif\.ssh\id_ed25519 -o StrictHostKeyChecking=no -tt hn05v8n20u7btj-6441183f@ssh.runpod.io

# Sync local changes to pod
python runpod_controller.py sync

# Install/update dependencies on pod
bash /workspace/FitFusion/runpod_setup.sh
```

**Pod Python environment:**

| Package | Version |
|---|---|
| torch | 2.2.0+cu121 |
| diffusers | 0.25.0 |
| transformers | 4.36.2 |
| huggingface_hub | 0.23.4 |
| numpy | 1.26.4 |
| accelerate | latest |
| CUDA | 12.1 |

**Known environment notes:**
- `huggingface_hub` must stay at `0.23.4` — newer versions break diffusers 0.25.0
- `numpy` must stay `< 2.0` — numpy 2.x breaks several downstream deps
- Weights symlinked: `IDM-VTON/ckpt` → `/workspace/FitFusion/ckpt`

---

## Key Design Decisions

### Texture-Preserving Mask Scaling
Size adaptation is applied **exclusively** to the agnostic boundary mask geometry. The garment image tensors remain at 1:1 aspect ratio. This prevents texture hallucination and preserves high-frequency details (logos, prints, patterns).

### Prompt Stripping
Diffusion prompts carry only pure garment identity (color, material, style). Physics descriptors like "tight", "stretching", "loose" are completely stripped. Fit conditioning is driven entirely by explicit mask geometry — not text conditionals.

### Mask Confidence Gate
OpenPose joints (Shoulder → Elbow → Wrist) define a structural arm bounding plane. Any SCHP mask proposing garment pixels outside this plane triggers a confidence penalty. Scores below 0.85 halt execution before diffusion to prevent generative waste. A `None`-type failsafe handles cropped photos where wrists/elbows are out of frame.

### Redis Concurrency (Legacy `tryon_api.py`)
The Celery/Redis broker in `IDM-VTON/tryon_api.py` prevents simultaneous API requests from causing GPU OOM crashes, isolating tensor processing from binary Pickle conflicts.

---

## Training

IDM-VTON QLoRA fine-tuning scripts are available for further size-conditioning training:

```bash
# Full fine-tune
python IDM-VTON/train_xl.py

# QLoRA fine-tune (memory efficient)
python IDM-VTON/train_xl_qlora.py

# On RunPod
bash start_training.sh
```

See [MODEL_SELECTION_AND_TRAINING_PLAN.md](MODEL_SELECTION_AND_TRAINING_PLAN.md) for full model comparison (IDM-VTON vs CatVTON vs CatV2TON vs FitDiT) and the rationale for choosing IDM-VTON.

---

## Current Status

| Component | Status |
|---|---|
| Brand catalog system | Complete |
| Size physics engine | Complete |
| Background preprocessing | Complete |
| Mask confidence scoring | Complete |
| Size-adaptive mask scaling | Complete |
| Skin compositing | Complete |
| IDM-VTON model loader | Complete |
| Diffusion inference executor | Complete |
| `run_pipeline.py` entry point | Complete |
| OpenPose wrapper (`src/pose/openpose.py`) | Complete |
| DensePose wrapper (`src/pose/densepose.py`) | Complete |
| SCHP human parsing (`src/masking/human_parsing.py`) | Complete |
| VITON-HD test dataset on pod | Missing — required for full inference test |
| QLoRA training run | Not started |
