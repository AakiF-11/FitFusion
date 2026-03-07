# FitFusion — Size-Aware Virtual Try-On

**Production-grade Size-Conditioned Virtual Try-On pipeline built on IDM-VTON (SDXL)**

> GitHub: `https://github.com/AakiF-11/FitFusion` | Branch: `master`  
> Inference Platform: RunPod (NVIDIA RTX A6000, 48 GB VRAM)  
> Base Model: IDM-VTON (Stable Diffusion XL fine-tuned for try-on)

---

## Project Overview

FitFusion wraps IDM-VTON with a physics-informed size-aware pipeline. Instead of a generic one-size image overlay, it uses real garment measurements, brand size charts, and fabric physics to show exactly how a specific size will drape on a specific body.

The entry point is `IDM-VTON/run_tryon.py`. It handles everything: downloading the customer photo, matching their body to the best reference model, applying physics parameters, and running IDM-VTON inference — with optional all-sizes comparison generation.

---

## Repository Structure

```
FitFusion/
│
├── IDM-VTON/                        # All model code — entry point is here
│   ├── run_tryon.py                 # MAIN ENTRY POINT — brand catalog + physics + inference
│   ├── tryon_api.py                 # TryOnAPI: init_tryon / generate_tryon (Step 1 & 2)
│   ├── size_aware_vton.py           # SizeAwareVTON: 7-layer physics engine
│   ├── brand_catalog.py             # BrandCatalog: height-gated body matching
│   ├── size_charts.py               # Brand size chart data + compute_size_ratio()
│   ├── size_adaptive_mask.py        # Agnostic mask scaling logic
│   ├── size_aware_pipeline.py       # Garment resizer using physics ratios
│   ├── measurement_encoder.py       # Body measurement encoder
│   ├── tps_warper.py                # Thin-plate spline garment warper
│   ├── inference.py                 # IDM-VTON base inference script
│   ├── inference_dc.py              # IDM-VTON inference variant
│   ├── generate_densepose.py        # DensePose generation utility
│   ├── prepare_vitonhd_dataset.py   # VITON-HD dataset preparation
│   ├── size_evaluation.py           # Fit quality evaluation
│   ├── train_xl.py                  # IDM-VTON full fine-tuning
│   ├── train_xl_qlora.py            # IDM-VTON QLoRA fine-tuning
│   ├── src/                         # IDM-VTON model source (UNet, pipeline, attention)
│   │   ├── unet_hacked_tryon.py
│   │   ├── unet_hacked_garmnet.py
│   │   ├── tryon_pipeline.py
│   │   └── ...
│   ├── ckpt/                        # Model weights (gitignored)
│   │   ├── unet/
│   │   ├── unet_encoder/
│   │   ├── vae/
│   │   ├── text_encoder/ text_encoder_2/
│   │   ├── image_encoder/
│   │   ├── scheduler/
│   │   ├── densepose/
│   │   ├── humanparsing/
│   │   └── openpose/
│   └── gradio_demo/
│
├── fitfusion/                       # Shared utilities imported by the pipeline
│   ├── utils/preprocessing.py       # RMBG-1.4 background removal + neckline erase
│   └── masking/
│       ├── compositor.py            # Skin/tattoo restoration (cv2.addWeighted)
│       └── confidence_scorer.py     # Arm-geometry mask confidence gate
│
├── data/
│   ├── brand_catalog/               # Brand JSON manifests + product images
│   │   ├── catalog.json
│   │   ├── snag_tights/
│   │   └── universal_standard/
│   ├── measurements/
│   │   ├── models.json              # Reference model body measurements
│   │   └── garments.json            # Garment size chart measurements per brand
│   └── test_inputs/
│       ├── person_model_C_Pixie.jpg
│       ├── garment_cropped_borg_aviator_jacket_black.png
│       └── test_config.json
│
├── training_data_extraction/        # Raw scraped brand data
│   ├── snag_tights/
│   └── universal_standard/
│
├── tests/
│   └── evaluation_matrix.py
│
├── runpod_controller.py             # Pod lifecycle (start/stop/sync)
├── runpod_setup.sh                  # Pod dependency installer
├── run_inference_pod.sh             # Pod inference runner (correct entry point)
├── start_training.sh                # QLoRA training launcher
├── extract_snag_tights.py           # Brand data extraction
├── extract_universal_standard.py    # Brand data extraction
├── requirements.txt
├── FitFusion_README.md              # Detailed architecture narrative
└── MODEL_SELECTION_AND_TRAINING_PLAN.md
```

---

## Architecture

```
Customer photo + measurements + product_id
                   │
                   ▼
         BrandCatalog.match_customer()
         → Height-gated Euclidean body match
         → Returns closest reference model + size
                   │
                   ▼
         TryOnAPI.generate_tryon()
         → Loads catalog garment for selected size
         → Calls SizeAwareVTON.prepare_size_aware_inputs()
                   │
                   ▼
         SizeAwareVTON (7-layer physics)
         → classify_fit() → FitProfile (warp, tension, mask_expansion)
         → Geometry-scale agnostic mask (mask-only, not texture)
         → Compute prompts (fit physics stripped from text)
         → Save: garment_preprocessed.png, mask_adapted.png, person_resized.png
                   │
                   ▼
         Celery task: run_inference()
         → DensePose extraction (inline detectron2)
         → IDM-VTON diffusion pipeline
         → SDXL inpainting at 1024×768
                   │
                   ▼
         Skin compositing
         → Restore face, neck, arms, legs over generated result
         → Output: result.png
```

---

## Quick Start

```bash
# Single size try-on
cd IDM-VTON
python run_tryon.py \
    --product_id "cropped-borg-aviator-jacket-black" \
    --brand_id   "snag_tights" \
    --s3_image_url "file:///workspace/FitFusion/data/test_inputs/person_model_C_Pixie.jpg" \
    --bust 92 --waist 76 --hips 100 --height 167 \
    --size XL \
    --output /workspace/output_XL.png \
    --steps 30 --seed 42

# All sizes — generates one image per size + comparison table
python run_tryon.py \
    --product_id "cropped-borg-aviator-jacket-black" \
    --brand_id   "snag_tights" \
    --s3_image_url "file:///workspace/FitFusion/data/test_inputs/person_model_C_Pixie.jpg" \
    --bust 92 --waist 76 --hips 100 --height 167 \
    --all_sizes \
    --output /workspace/output_all_sizes.png \
    --steps 30 --seed 42
```

### CLI Arguments

| Argument | Description |
|---|---|
| `--product_id` | Product ID from the brand catalog |
| `--brand_id` | Brand identifier (`snag_tights`, `universal_standard`) |
| `--s3_image_url` | Customer photo: `s3://bucket/photo.jpg` or `file:///path/photo.jpg` |
| `--bust` / `--waist` / `--hips` | Customer measurements in cm |
| `--height` | Customer height in cm (default: `170`) |
| `--size` | Target size to generate (e.g. `XL`, `M`, `3XL`) |
| `--all_sizes` | Generate all available sizes and print comparison table |
| `--output` | Output path for the result image |
| `--steps` | Diffusion steps (default: `30`) |
| `--seed` | Reproducibility seed (default: `42`) |
| `--no_model` | Preprocessing-only dry run (skip inference) |

---

## Checkpoint Layout

Weights live under `IDM-VTON/ckpt/` (gitignored). On the pod: `/workspace/FitFusion/IDM-VTON/ckpt/`.

```
ckpt/
├── unet/              # IDM-VTON fine-tuned UNet (SDXL)
├── unet_encoder/      # Garment feature encoder UNet
├── vae/               # Stable Diffusion XL VAE
├── text_encoder/      # CLIP ViT-L/14
├── text_encoder_2/    # CLIP ViT-bigG/14
├── tokenizer/ tokenizer_2/      # CLIP tokenizers
├── image_encoder/     # IP-Adapter image encoder
├── scheduler/         # DDPM noise scheduler
├── densepose/         # Detectron2 DensePose weights
├── humanparsing/      # SCHP (parsing_atr.onnx, parsing_lip.onnx)
└── openpose/          # OpenPose body_pose_model.pth
```

---

## Fit Classes

The physics engine maps `(person_size, garment_size)` to one of 7 fit classes:

| Class | Gap | Description |
|---|---|---|
| `VERY_TIGHT` | -3 | 3+ sizes smaller than body |
| `TIGHT` | -2 | 2 sizes smaller |
| `SNUG` | -1 | 1 size smaller |
| `STANDARD` | 0 | Matches body size |
| `RELAXED` | +1 | 1 size larger |
| `LOOSE` | +2 | 2 sizes larger |
| `OVERSIZED` | +3 | 3+ sizes larger |

Each class produces distinct `warp_intensity`, `mask_expansion_px`, `inpainting_strength`, and `edge_softness` values.

---

## Key Design Decisions

### Texture-Preserving Mask Scaling
Size adaptation is applied **only** to the agnostic mask geometry. The garment image tensor stays at 1:1 pixel ratio — preventing texture hallucination and preserving logos/prints.

### Prompt Stripping
Diffusion prompts carry only pure garment identity (color, material, style). Physics descriptors ("tight", "stretching", "loose") are stripped entirely. Fit conditioning is driven by explicit mask geometry, not text conditionals.

### Mask Confidence Gate (`fitfusion/masking/confidence_scorer.py`)
OpenPose joints define a structural arm bounding plane. Any mask proposing garment pixels outside this plane triggers a confidence penalty. Scores below 0.85 halt execution before diffusion.

### Redis Concurrency
The Celery/Redis broker prevents simultaneous API requests from causing GPU OOM, isolating tensor processing from binary Pickle failures.

---

## RunPod Deployment

**Pod:** `hn05v8n20u7btj-6441183f@ssh.runpod.io`  
**GPU:** NVIDIA RTX A6000 (48 GB VRAM)

```bash
# Connect
ssh -i C:\Users\Aakif\.ssh\id_ed25519 -o StrictHostKeyChecking=no -tt hn05v8n20u7btj-6441183f@ssh.runpod.io

# Sync local changes to pod
python runpod_controller.py sync

# Install/update dependencies on pod
bash /workspace/FitFusion/runpod_setup.sh

# Run inference
bash /workspace/FitFusion/run_inference_pod.sh
```

---

## Training

```bash
# QLoRA fine-tune (memory efficient — recommended)
python IDM-VTON/train_xl_qlora.py

# Full fine-tune
python IDM-VTON/train_xl.py

# On RunPod
bash start_training.sh
```

See [MODEL_SELECTION_AND_TRAINING_PLAN.md](MODEL_SELECTION_AND_TRAINING_PLAN.md) for model comparison.

---

## Current Status

| Component | Status |
|---|---|
| Brand catalog + body matching | Complete |
| Size physics engine (7 layers) | Complete |
| RMBG-1.4 background removal | Complete |
| Mask confidence scoring | Complete |
| Size-adaptive mask scaling | Complete |
| Skin/tattoo compositing | Complete |
| TryOnAPI (init + generate) | Complete |
| IDM-VTON diffusion inference | Complete |
| All-sizes comparison generation | Complete |
| S3 / local photo download layer | Complete |
| QLoRA training | Not started |

