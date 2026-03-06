"""
FitFusion — Size-Aware IDM-VTON Inference
==========================================
End-to-end inference script that uses the physics-based pipeline
to produce size-correct virtual try-on images.

This script:
1. Loads the pre-trained IDM-VTON model (no custom training needed)
2. Takes a person image + flat garment image + size labels
3. Uses physics to resize the garment based on size ratios
4. Runs IDM-VTON inference with the modified inputs
5. Outputs a size-correct try-on image

Usage:
    python infer_size_aware.py \
        --person_image person.jpg \
        --garment_image garment.jpg \
        --person_size M \
        --garment_size XL \
        --garment_type top \
        --output output.png
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from size_aware_pipeline import SizeAwarePipeline
from size_charts import compute_size_ratio


def load_idm_vton_pipeline(
    model_path: str = "yisol/IDM-VTON",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Load the pre-trained IDM-VTON pipeline.
    
    This uses the ORIGINAL IDM-VTON weights — no custom training needed.
    Size-awareness comes from our physics-based input preprocessing.
    """
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        AutoProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )
    
    # These imports depend on the IDM-VTON codebase
    try:
        from tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from unet_hacked_tryon import UNet2DConditionModel
    except ImportError:
        print("ERROR: IDM-VTON codebase not found in current directory.")
        print("Please run this script from the IDM-VTON directory.")
        sys.exit(1)
    
    print("[1/5] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).to(device)
    
    print("[2/5] Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=dtype
    ).to(device)
    
    print("[3/5] Loading text encoders...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    
    print("[4/5] Loading image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=dtype
    ).to(device)
    
    print("[5/5] Loading garment encoder (UNet encoder)...")
    unet_encoder = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet_encoder", torch_dtype=dtype
    ).to(device)
    
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    ip_processor = AutoProcessor.from_pretrained(model_path, subfolder="image_encoder")
    
    pipe = TryonPipeline.from_pretrained(
        model_path,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        image_encoder=image_encoder,
        scheduler=noise_scheduler,
        torch_dtype=dtype,
    ).to(device)
    
    return pipe, unet_encoder, ip_processor


def generate_agnostic_mask(person_image: Image.Image) -> Image.Image:
    """
    Generate an agnostic mask from a person image.
    Uses simple body detection to create the upper-body mask region.
    
    For production, this should be replaced with a proper human parser
    (e.g., SCHP, segformer).
    """
    img = np.array(person_image.convert("RGB"))
    h, w = img.shape[:2]
    
    # Simple heuristic: mask the upper torso region
    # (center 60% width, top 30% to 70% height)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    y1 = int(h * 0.15)
    y2 = int(h * 0.75)
    
    mask[y1:y2, x1:x2] = 255
    
    return Image.fromarray(mask)


def run_size_aware_inference(
    person_image_path: str,
    garment_image_path: str,
    person_size: str = "M",
    garment_size: str = "M",
    garment_type: str = "top",
    fit_preference: str = "regular",
    output_path: str = "output_size_aware.png",
    model_path: str = "yisol/IDM-VTON",
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Full end-to-end size-aware virtual try-on inference.
    """
    print("=" * 60)
    print("  FitFusion — Size-Aware Virtual Try-On")
    print("=" * 60)
    
    # 1. Load input images
    print(f"\nPerson image: {person_image_path}")
    print(f"Garment image: {garment_image_path}")
    print(f"Person size: {person_size}, Garment size: {garment_size}")
    print(f"Garment type: {garment_type}, Fit: {fit_preference}")
    
    person_img = Image.open(person_image_path).convert("RGB")
    garment_img = Image.open(garment_image_path).convert("RGB")
    
    # 2. Generate agnostic mask
    print("\n[Step 1] Generating agnostic mask...")
    agnostic_mask = generate_agnostic_mask(person_img)
    
    # 3. Apply physics-based size preprocessing
    print("[Step 2] Applying physics-based size preprocessing...")
    pipeline = SizeAwarePipeline(target_resolution=(768, 1024))
    
    result = pipeline.prepare_inputs(
        garment_image=garment_img,
        agnostic_mask=agnostic_mask,
        garment_type=garment_type,
        garment_size=garment_size,
        person_size=person_size,
        fit_preference=fit_preference,
    )
    
    resized_garment = result["garment_image"]
    adapted_mask = result["agnostic_mask"]
    size_info = result["size_info"]
    
    print(f"  Width ratio: {size_info['width_ratio']:.3f}x "
          f"({size_info['garment_width_cm']}cm / {size_info['person_width_cm']}cm)")
    print(f"  Length ratio: {size_info['length_ratio']:.3f}x")
    print(f"  Size gap: {size_info['size_gap']:+d}")
    
    # Save intermediate files for debugging
    debug_dir = Path(output_path).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    resized_garment.save(str(debug_dir / "resized_garment.png"))
    adapted_mask.save(str(debug_dir / "adapted_mask.png"))
    print(f"  Debug files saved to {debug_dir}/")
    
    # 4. Load IDM-VTON model
    print("\n[Step 3] Loading IDM-VTON model...")
    try:
        pipe, unet_encoder, ip_processor = load_idm_vton_pipeline(
            model_path=model_path, device=device
        )
    except Exception as e:
        print(f"\nCould not load IDM-VTON model: {e}")
        print("Saving preprocessed inputs only (no model inference).")
        
        # Save a comparison showing the preprocessing
        comparison = pipeline.visualize_size_comparison(
            garment_img, agnostic_mask, garment_type, person_size,
            ["S", "M", "L", "XL", "2XL"]
        )
        comparison.save(output_path.replace(".png", "_comparison.png"))
        print(f"Saved comparison to {output_path.replace('.png', '_comparison.png')}")
        return
    
    # 5. Resize person image to model input size
    person_resized = person_img.resize((768, 1024), Image.LANCZOS)
    
    # 6. Process garment through IP-Adapter
    print("[Step 4] Processing garment through encoder...")
    ip_image = ip_processor(images=resized_garment, return_tensors="pt")
    
    # 7. Create agnostic image (person with masked region)
    person_np = np.array(person_resized)
    mask_np = np.array(adapted_mask.resize((768, 1024), Image.NEAREST))
    agnostic_img_np = person_np.copy()
    agnostic_img_np[mask_np > 128] = [128, 128, 128]  # Gray out masked region
    agnostic_img = Image.fromarray(agnostic_img_np)
    
    # 8. Run IDM-VTON inference
    print(f"[Step 5] Running inference ({num_inference_steps} steps)...")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        output = pipe(
            prompt="a photo of a person wearing a garment",
            negative_prompt="low quality, bad anatomy, deformed",
            image=person_resized,
            mask_image=adapted_mask,
            ip_adapter_image=resized_garment,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    
    result_image = output.images[0]
    result_image.save(output_path)
    print(f"\n✓ Size-aware try-on saved to {output_path}")
    
    return result_image


def generate_multi_size_comparison(
    person_image_path: str,
    garment_image_path: str,
    person_size: str = "M",
    garment_type: str = "top",
    sizes: list = None,
    output_path: str = "multi_size_comparison.png",
):
    """
    Generate a visual comparison of the PREPROCESSED garment across sizes.
    This doesn't need the model — it shows what the physics pipeline does
    to the garment image before it enters the model.
    """
    if sizes is None:
        sizes = ["XS", "S", "M", "L", "XL", "2XL"]
    
    garment_img = Image.open(garment_image_path).convert("RGB")
    
    # Simple agnostic mask
    person_img = Image.open(person_image_path).convert("RGB")
    agnostic_mask = generate_agnostic_mask(person_img)
    
    pipeline = SizeAwarePipeline(target_resolution=(768, 1024))
    
    comparison = pipeline.visualize_size_comparison(
        garment_img, agnostic_mask, garment_type, person_size, sizes
    )
    comparison.save(output_path)
    print(f"Multi-size comparison saved to {output_path}")
    
    # Also print the ratios
    print(f"\nSize ratios (person={person_size}):")
    for gs in sizes:
        ratio = compute_size_ratio(garment_type, gs, person_size)
        print(f"  {gs:>4s}: W={ratio['width_ratio']:.3f}x  L={ratio['length_ratio']:.3f}x  gap={ratio['size_gap']:+d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Size-Aware Virtual Try-On")
    parser.add_argument("--person_image", type=str, required=True)
    parser.add_argument("--garment_image", type=str, required=True)
    parser.add_argument("--person_size", type=str, default="M")
    parser.add_argument("--garment_size", type=str, default="M")
    parser.add_argument("--garment_type", type=str, default="top")
    parser.add_argument("--fit", type=str, default="regular",
                        choices=["tight", "regular", "loose", "oversized"])
    parser.add_argument("--output", type=str, default="output_size_aware.png")
    parser.add_argument("--model_path", type=str, default="yisol/IDM-VTON")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_only", action="store_true",
                        help="Only show preprocessing comparison, no model inference")
    
    args = parser.parse_args()
    
    if args.compare_only:
        generate_multi_size_comparison(
            person_image_path=args.person_image,
            garment_image_path=args.garment_image,
            person_size=args.person_size,
            garment_type=args.garment_type,
            output_path=args.output,
        )
    else:
        run_size_aware_inference(
            person_image_path=args.person_image,
            garment_image_path=args.garment_image,
            person_size=args.person_size,
            garment_size=args.garment_size,
            garment_type=args.garment_type,
            fit_preference=args.fit,
            output_path=args.output,
            model_path=args.model_path,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
