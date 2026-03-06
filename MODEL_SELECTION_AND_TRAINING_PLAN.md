# Size-Aware VTON: Model Selection & QLoRA Training Plan

## 1. Model Comparison

| Criteria | IDM-VTON | CatVTON | CatV2TON | FitDiT |
|---|---|---|---|---|
| **Paper** | ECCV 2024 | ICLR 2025 | CVPR 2025 Workshop | - |
| **Stars** | 4,900 | 1,600 | 195 | ~200 |
| **Base Model** | SDXL Inpainting | SD 1.5 Inpainting | EasyAnimate (DiT) | SD3 (DiT) |
| **Total Params** | ~3.5B (UNet) + IP-Adapter | 899M | Unknown (DiT) | ~4-5B |
| **Trainable Params** | Full UNet (all) | 49.57M | Unknown | N/A |
| **Training Code** | **YES** (`train_xl.py`, 797 lines) | NO (inference only) | NO (inference only) | NO |
| **LoRA/QLoRA Ready** | YES (diffusers UNet + PEFT) | Partial (SD1.5 base) | NO | NO |
| **Conditioning** | IP-Adapter cross-attention | Concatenation | Temporal concat | Dual-DiT attention |
| **Resolution** | 1024×768 | 1024×768 | 256/512 | 1024×768 |
| **Inference VRAM** | ~12-16 GB | < 8 GB | Unknown | ~24 GB |
| **Dataset Format** | VITON-HD / DressCode | VITON-HD / DressCode | VITON-HD / DressCode | Custom |
| **License** | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | Unknown | CC BY-NC-SA 4.0 |

---

## 2. Recommendation: IDM-VTON

**IDM-VTON is the clear winner** for size-aware QLoRA fine-tuning. Here's why:

### Why IDM-VTON?

1. **Only model with published training code** — `train_xl.py` is 797 lines of production-ready training with `accelerate`, gradient checkpointing, 8-bit Adam, mixed precision, SNR weighting, and checkpoint saving. No other VTON model provides this.

2. **Industry standard** — 4,900 GitHub stars, 799 forks, ECCV 2024. The most widely adopted open-source VTON model.

3. **IP-Adapter cross-attention = natural size injection point** — The IP-Adapter injects garment features via cross-attention into the UNet. We can inject body measurement embeddings the same way, adding a `MeasurementEncoder` alongside the existing `ImageProjection`. This is architecturally clean.

4. **SDXL quality** — Generates at 1024×768 with high fidelity. SDXL is the most mature and well-supported diffusion architecture in the ecosystem.

5. **Already supports memory-efficient training** — `--use_8bit_adam`, `--gradient_checkpointing`, `--mixed_precision bf16` are built in. Adding QLoRA via PEFT is a small modification.

6. **Diffusers-native** — Built on HuggingFace diffusers, which has first-class PEFT/LoRA integration. The UNet inherits `PeftAdapterMixin`.

### Why NOT the others?

- **CatVTON**: Lightest model (49.57M trainable), but NO training code published. Its concatenation-based conditioning is simpler but harder to inject size info into (no cross-attention mechanism). Would need to reverse-engineer the entire training pipeline.
  
- **CatV2TON**: Newest (CVPR 2025 Workshop) but only 195 stars, NO training code, and early-stage with limited community. EasyAnimate DiT base is less mature.

- **FitDiT**: Already in our workspace but has NO training code, no LoRA adapters defined despite `PeftAdapterMixin` inheritance, and custom SD3 architecture that's harder to adapt.

---

## 3. QLoRA Training Architecture

### 3.1 How IDM-VTON Works (Baseline)

```
Input: person_image + garment_image + densepose + agnostic_mask + text_prompt

┌─────────────────────────────────────────────────────────┐
│                    IDM-VTON Pipeline                     │
│                                                         │
│  garment_image ──→ CLIP Image Encoder ──→ IP-Adapter    │
│                         │                    │          │
│                    Image Projection      Cross-Attn     │
│                    (Resampler)           injection       │
│                         │                    │          │
│  garment_image ──→ UNet_Encoder ──→ Reference Features  │
│                                          │              │
│  person + mask + densepose ──→ UNet (Denoiser) ←──┘     │
│                                    │                    │
│                              VAE Decode                 │
│                                    │                    │
│                            Output Image                 │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: The UNet receives conditioning from TWO paths:
1. **IP-Adapter path**: CLIP features → Resampler → cross-attention (semantic garment understanding)
2. **Reference path**: UNet_Encoder → reference features → self-attention (spatial garment details)

### 3.2 Size-Aware Modification

We inject body measurements as an ADDITIONAL conditioning signal:

```
NEW: body_measurements ──→ MeasurementEncoder ──→ Size Embedding
                                                       │
                                                  Cross-Attn
                                                  injection
                                                       │
            garment + person + mask + densepose ──→ UNet ←──┘
```

**MeasurementEncoder** (new, small, fully trainable):
```python
class MeasurementEncoder(nn.Module):
    """Encodes body measurements into conditioning embeddings."""
    def __init__(self, num_measurements=6, embed_dim=1280, num_tokens=4):
        super().__init__()
        # Input: [height, bust, waist, hips, size_category, garment_type]
        self.mlp = nn.Sequential(
            nn.Linear(num_measurements, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, embed_dim * num_tokens),
        )
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
    
    def forward(self, measurements):
        # measurements: (B, num_measurements)
        x = self.mlp(measurements)  # (B, embed_dim * num_tokens)
        return x.reshape(-1, self.num_tokens, self.embed_dim)
```

**Total new trainable params**: ~1.3M (MeasurementEncoder) + ~50-100M (LoRA adapters) = **~51-101M params**

### 3.3 QLoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # scaling factor
    target_modules=[
        "to_q", "to_k", "to_v",   # self-attention
        "to_out.0",                # output projection
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=None,                # not a specific task
)

# Quantize UNet to 4-bit
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Apply LoRA to UNet
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()
# Expected: trainable params: ~50M || all params: ~3.5B || trainable%: 1.43%
```

### 3.4 What Gets Trained vs Frozen

| Component | Status | Params |
|---|---|---|
| VAE (encoder/decoder) | FROZEN | ~84M |
| CLIP Text Encoders (x2) | FROZEN | ~500M |
| CLIP Image Encoder | FROZEN | ~632M |
| UNet (base weights) | FROZEN (4-bit quantized) | ~3.5B → ~1.75GB |
| UNet LoRA adapters | **TRAINED** | ~50M |
| IP-Adapter (Resampler) | **TRAINED** | ~10M |
| UNet_Encoder (garment) | FROZEN or partial LoRA | ~800M |
| **MeasurementEncoder** | **TRAINED (new)** | ~1.3M |
| **Total trainable** | | **~61M** |

---

## 4. Data Pipeline: Converting Our Data to VITON-HD Format

### 4.1 Required Dataset Structure

IDM-VTON expects the VITON-HD format:
```
dataset/
├── train_pairs.txt          # "image_name cloth_name" per line
├── train/
│   ├── image/               # person wearing the garment (target)
│   │   ├── 00001_00.jpg
│   ├── cloth/               # flat/isolated garment image
│   │   ├── 00001_00.jpg
│   ├── agnostic-mask/       # binary mask of clothing region
│   │   ├── 00001_00_mask.png
│   ├── image-densepose/     # DensePose of the person
│   │   ├── 00001_00.jpg
│   └── measurements.json    # NEW: body measurements per image
```

### 4.2 Converting Snag Tights Data

Our Snag Tights data has the structure: `ModelName-Size-ProductName_ShotNumber.jpg`

Conversion plan:
1. **image/** ← Model photo (person wearing the garment)
2. **cloth/** ← Use the *product catalog image* (not the model shot) OR generate a flat garment image by removing the model. Since Snag Tights has on-model shots only, we'll use one size's photo as the "cloth reference" and another size's photo as the "target"
3. **agnostic-mask/** ← Generate using SCHP (Self-Correction Human Parsing) or DensePose segmentation — both supported by IDM-VTON
4. **image-densepose/** ← Generate using DensePose (Detectron2-based)
5. **measurements.json** ← From Snag's size chart: `{size: {hip_cm, waist_cm, height_cm}}`

### 4.3 Data Pairing Strategy (Size-Aware)

For size-aware training, we create pairs where the SAME garment appears on DIFFERENT sized models:

```
# Training pairs with size conditioning:
# Format: target_image  cloth_ref_image  target_measurements

# Same garment, different sizes → model learns size variation
model_A_size_S_legging_001.jpg    model_B_size_XL_legging_001.jpg    {waist: 66, hips: 91}
model_B_size_XL_legging_001.jpg   model_A_size_S_legging_001.jpg     {waist: 96, hips: 122}

# This teaches the model:
# "Given this garment on person X, show it on a person with measurements Y"
```

### 4.4 Data Augmentation

From our ~2,700 paired images across 287 products:
- **Cross-size pairing**: For products with N sizes, we get N×(N-1) pairs → ~5,000-8,000 training pairs
- **Horizontal flip**: 2× multiplier → ~10,000-16,000 effective pairs  
- **Color jitter**: Already in IDM-VTON's training code
- **Random crop/scale**: Already in IDM-VTON's training code

**Estimated effective training set: ~10,000-16,000 pairs** — sufficient for QLoRA fine-tuning.

---

## 5. RunPod GPU Recommendation

### 5.1 Pricing (Community Cloud, Pods)

| GPU | VRAM | Price/hr | Best For |
|---|---|---|---|
| RTX A5000 | 24 GB | $0.27/hr | Budget option (tight for SDXL) |
| L4 | 24 GB | $0.39/hr | Good value, limited VRAM |
| **A40** | **48 GB** | **$0.40/hr** | **Best value — recommended** |
| RTX 3090 | 24 GB | $0.46/hr | Consumer GPU, decent |
| **RTX A6000** | **48 GB** | **$0.49/hr** | **Premium pick — great VRAM** |
| RTX 4090 | 24 GB | $0.59/hr | Fast but tight VRAM |
| A100 PCIe | 80 GB | $1.39/hr | Overkill for QLoRA |

### 5.2 Recommendation: A40 48GB ($0.40/hr)

- **48 GB VRAM** — plenty of headroom for QLoRA with SDXL (estimated ~18-22 GB usage)
- **$0.40/hr** — cheapest 48 GB option on RunPod
- Can comfortably run batch_size=4 with gradient accumulation
- Fallback: RTX A6000 ($0.49/hr) if A40 availability is low

### 5.3 Estimated Training Cost

| Parameter | Estimate |
|---|---|
| Training pairs | ~10,000-16,000 (with augmentation) |
| Batch size | 4 (with gradient accumulation of 4 = effective 16) |
| Epochs | 10-20 |
| Steps per epoch | ~625-1,000 |
| Time per step | ~2-3 seconds (QLoRA on A40) |
| Total training time | ~10-20 hours |
| **GPU cost** | **$4-8** |
| Storage (50 GB) | ~$2.50/mo |
| **Total estimated cost** | **$7-11** |

---

## 6. Implementation Roadmap

### Phase 1: Data Preprocessing (Local, ~2 days)
1. Install DensePose (Detectron2) locally or use lightweight alternative
2. Generate agnostic masks for all Snag Tights model images
3. Generate DensePose maps for all model images
4. Build cross-size pairing logic
5. Create measurements.json from Snag size chart
6. Validate dataset structure matches VITON-HD format

### Phase 2: Code Modification (Local, ~1-2 days)
1. Clone IDM-VTON repository
2. Modify `train_xl.py` to add QLoRA support:
   - Add `BitsAndBytesConfig` for 4-bit quantization
   - Add `LoraConfig` and wrap UNet with `get_peft_model()`
   - Add `MeasurementEncoder` module
   - Modify `VitonHDDataset` to load measurements
   - Inject measurement embeddings into cross-attention
3. Add size-aware inference pipeline
4. Test locally with 1-2 samples (dry run, CPU/small batch)

### Phase 3: Cloud Training (RunPod, ~1 day)
1. Create RunPod pod with A40 48GB
2. Upload dataset + modified code
3. Run training: `accelerate launch train_xl_qlora.py ...`
4. Monitor loss curves and sample outputs
5. Save checkpoints every 2 epochs
6. Download best checkpoint

### Phase 4: Integration & Testing (Local, ~1 day)  
1. Load QLoRA adapter weights into IDM-VTON pipeline
2. Build size-aware inference: input (person photo, garment, target measurements) → output
3. Test with various body types and garment sizes
4. Integrate into FitFusion UI

---

## 7. Key Files to Create

```
FitFusion/
├── idm_vton/                          # IDM-VTON with QLoRA mods
│   ├── train_xl_qlora.py              # Modified training script
│   ├── measurement_encoder.py         # New: MeasurementEncoder
│   ├── dataset_size_aware.py          # New: Size-aware dataset loader
│   ├── inference_size_aware.py        # New: Size-aware inference
│   └── config/
│       ├── qlora_config.yaml          # LoRA hyperparameters
│       └── training_config.yaml       # Training hyperparameters
├── data_pipeline/
│   ├── convert_snag_to_vitonhd.py     # Convert Snag data → VITON-HD format
│   ├── generate_densepose.py          # Generate DensePose maps
│   ├── generate_agnostic_mask.py      # Generate agnostic masks
│   └── create_training_pairs.py       # Cross-size pairing logic
└── runpod/
    ├── setup_runpod.sh                # Pod setup script
    ├── upload_data.sh                 # Upload dataset to pod
    └── start_training.sh             # Launch training
```

---

## 8. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Snag Tights images are on-model, not flat garment | Use cross-size pairing (one model's photo as cloth reference for another) |
| DensePose generation may fail on some poses | Use fallback: OpenPose or manual mask correction |
| 2,700 images may not be enough diversity | Heavy augmentation + cross-size pairs → ~10-16k effective pairs |
| QLoRA may degrade base VTON quality | Use low rank (r=16), careful learning rate (1e-5), and extensive validation |
| A40 GPU may not be available on RunPod | Fallback: RTX A6000 ($0.49/hr) or RTX 4090 with smaller batch |
| Measurement encoder may not learn useful features | Pre-train measurement encoder on synthetic data first |

---

## Summary

- **Model**: IDM-VTON (ECCV 2024, 4.9k stars, only model with training code)
- **Method**: QLoRA (4-bit quantization + LoRA rank 16 + MeasurementEncoder)
- **GPU**: RunPod A40 48GB at $0.40/hr
- **Data**: ~2,700 paired images → ~10-16k training pairs with augmentation
- **Cost**: ~$7-11 total
- **Timeline**: ~5-6 days from start to working prototype
- **Trainable params**: ~61M (vs 3.5B total = 1.7% of model)
