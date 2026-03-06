"""
src/pose/densepose.py
=====================
Wrapper for FitFusion's segformer-based DensePose-like IUV generation.

Since Detectron2 cannot be installed cleanly on all CUDA environments,
IDM-VTON/generate_densepose.py implements a DensePose *approximation* using
``mattmdjaga/segformer_b2_clothes`` — a HuggingFace SegFormer model that
maps clothing/body region labels to DensePose IUV channel values.

IUV format (3-channel uint8 image, same spatial resolution as input):
    Ch 0 (I) — body-part index × 10 (0–255), where 0 = background
    Ch 1 (U) — normalised horizontal position within that body part (0–255)
    Ch 2 (V) — normalised vertical position within that body part (0–255)

Upstream code:  IDM-VTON/generate_densepose.py
Model:          mattmdjaga/segformer_b2_clothes
                Loaded from <ckpt_dir> if a local HF snapshot is present,
                otherwise downloaded from HuggingFace Hub.
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union

log = logging.getLogger(__name__)

# ── IDM-VTON path setup ───────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parents[2]
_IDM_DIR = _ROOT / "IDM-VTON"
if str(_IDM_DIR) not in sys.path:
    sys.path.insert(0, str(_IDM_DIR))

# ── HuggingFace model identifier ─────────────────────────────────────────────
_HF_MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# ── Module-level singleton cache ──────────────────────────────────────────────
_segformer_cache: Dict[str, Any] = {}   # ckpt_dir → (processor, model, device)


def _get_segformer(ckpt_dir: str):
    """
    Lazy-load and cache the SegFormer segmentation model.

    Load order:
        1. ``<ckpt_dir>`` — treated as a local HuggingFace model snapshot if
           ``config.json`` is present (supports fully offline RunPod pods).
        2. ``mattmdjaga/segformer_b2_clothes`` from HuggingFace Hub.
    """
    if ckpt_dir in _segformer_cache:
        return _segformer_cache[ckpt_dir]

    import torch
    from transformers import (
        SegformerImageProcessor,
        SegformerForSemanticSegmentation,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_config = Path(ckpt_dir) / "config.json"
    model_source = str(Path(ckpt_dir)) if local_config.exists() else _HF_MODEL_ID
    log.info("Loading DensePose segformer from '%s' on %s …", model_source, device)

    processor = SegformerImageProcessor.from_pretrained(model_source)
    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(model_source)
        .to(device)
        .eval()
    )

    _segformer_cache[ckpt_dir] = (processor, model, device)
    log.info("DensePose segformer ready.")
    return processor, model, device


# ── Public API ────────────────────────────────────────────────────────────────
def generate_densepose_map(
    image_path: Union[str, np.ndarray, Any],
    ckpt_dir: str,
    output_dir: Optional[str] = None,
) -> Any:
    """
    Generate a DensePose-like IUV map for the input person image.

    run_pipeline.py passes ``ckpt_dir = ./ckpt/densepose``.

    Args:
        image_path: Person image — file path (str), PIL Image, or HWC uint8
                    numpy array.
        ckpt_dir:   Path to ``ckpt/densepose/`` directory.  If it contains a
                    local HuggingFace snapshot (``config.json`` present) it is
                    loaded offline; otherwise the model is fetched from HF Hub.
        output_dir: If provided, the IUV PNG is saved here as
                    ``densepose_iuv.png``.

    Returns:
        iuv_image: PIL RGB Image (H × W, same size as input) where:
                   R = body-part index × 10,  G = U coord,  B = V coord.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage
    from generate_densepose import generate_iuv_from_segmentation

    processor, model, device = _get_segformer(ckpt_dir)

    # ── Normalise input to PIL RGB ─────────────────────────────────────────────
    if isinstance(image_path, str):
        pil_img = PILImage.open(image_path).convert("RGB")
    elif isinstance(image_path, np.ndarray):
        pil_img = PILImage.fromarray(image_path.astype(np.uint8)).convert("RGB")
    else:
        pil_img = image_path.convert("RGB")

    orig_w, orig_h = pil_img.size   # PIL returns (W, H)

    # ── Run SegFormer segmentation ────────────────────────────────────────────
    with torch.no_grad():
        inputs    = processor(images=[pil_img], return_tensors="pt").to(device)
        outputs   = model(**inputs)
        logits    = outputs.logits   # (1, num_labels, H/4, W/4)

        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )                            # (1, num_labels, H, W)
        seg_map = (
            upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H, W) int64
        )

    # ── Convert segmentation labels → DensePose IUV ───────────────────────────
    # generate_iuv_from_segmentation converts each segformer label to a
    # DensePose body-part index and fills normalised U/V coordinates.
    iuv = generate_iuv_from_segmentation(seg_map, orig_h, orig_w)   # (H, W, 3) uint8
    iuv_pil = PILImage.fromarray(iuv)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "densepose_iuv.png")
        iuv_pil.save(save_path)
        log.info("DensePose IUV saved → %s", save_path)

    return iuv_pil
