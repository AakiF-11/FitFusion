"""
src/inference/model_loader.py
Loads the full IDM-VTON pipeline from local ckpt/ directory.
No HuggingFace downloads — weights are already on disk.
"""
import sys
import torch
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))
sys.path.insert(0, str(_ROOT / "IDM-VTON" / "src"))
sys.path.insert(0, str(_ROOT / "IDM-VTON" / "gradio_demo"))


def load_idm_vton_pipeline(ckpt_dir: str, device: str = "cuda") -> Any:
    """
    Load IDM-VTON inference pipeline from local weights.
    Weights are expected to already exist at ckpt_dir (no download).

    Args:
        ckpt_dir: Absolute path to the ckpt/ directory.
        device:   "cuda" or "cpu".

    Returns:
        pipeline: Loaded TryonPipeline ready for inference.
    """
    # FitFusion's own 'src' package is already in sys.modules by Stage 3/4.
    # IDM-VTON's internal code also uses 'from src.X import' expecting its own
    # src/ dir.  Temporarily evict FitFusion's 'src' subtree so IDM-VTON's
    # imports resolve to IDM-VTON/src/ instead.
    _saved_src = {k: sys.modules.pop(k)
                  for k in list(sys.modules)
                  if k == "src" or k.startswith("src.")}
    try:
        from diffusers import AutoencoderKL
        from transformers import (
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            CLIPVisionModelWithProjection,
        )
        from tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from unet_hacked_tryon import UNet2DConditionModel
        from unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
    finally:
        sys.modules.update(_saved_src)

    dtype = torch.float16 if device == "cuda" else torch.float32

    unet = UNet2DConditionModel.from_pretrained(
        ckpt_dir, subfolder="unet", torch_dtype=dtype
    ).to(device)

    unet_encoder = GarmentUNet.from_pretrained(
        ckpt_dir, subfolder="unet_encoder", torch_dtype=dtype
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        ckpt_dir, subfolder="vae", torch_dtype=dtype
    ).to(device)

    text_encoder = CLIPTextModel.from_pretrained(
        ckpt_dir, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        ckpt_dir, subfolder="text_encoder_2", torch_dtype=dtype
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        ckpt_dir, subfolder="image_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(ckpt_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(ckpt_dir, subfolder="tokenizer_2")

    pipeline = TryonPipeline.from_pretrained(
        ckpt_dir,
        unet=unet,
        unet_encoder=unet_encoder,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        image_encoder=image_encoder,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=dtype,
    ).to(device)

    return pipeline
