"""
src/inference/tryon.py
Executes a single size-aware try-on pass through the IDM-VTON pipeline.
Accepts pre-processed inputs; returns a PIL Image.
"""
import torch
import numpy as np
from PIL import Image
from typing import Any


def run_tryon(
    pipeline: Any,
    person_image_path: str,
    warped_garment: Image.Image,
    agnostic_mask: Image.Image,
    densepose_map: Image.Image,
    physics_params: Any,
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    width: int = 768,
    height: int = 1024,
    seed: int = 42,
) -> Image.Image:
    """
    Run IDM-VTON inference with size-aware inputs.

    Args:
        pipeline:            Loaded TryonPipeline from model_loader.
        person_image_path:   Path to standardized person image.
        warped_garment:      Garment PIL Image after size physics applied.
        agnostic_mask:       Inpainting mask (white = region to fill).
        densepose_map:       DensePose UV PIL Image.
        physics_params:      PhysicsParams dataclass from SizeAwareVTON.
        num_inference_steps: Diffusion steps.
        guidance_scale:      CFG scale.
        width, height:       Output resolution.
        seed:                RNG seed for reproducibility.

    Returns:
        result: PIL Image of the final try-on.
    """
    person_image = Image.open(person_image_path).convert("RGB").resize((width, height))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.no_grad():
        result = pipeline(
            prompt=physics_params.positive_prompt,
            image=person_image,
            mask_image=agnostic_mask,
            garment_image=warped_garment,
            pose_image=densepose_map,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

    return result
