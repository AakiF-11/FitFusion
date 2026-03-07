"""
FitFusion — Unified RunPod Inference Script
=============================================
The complete end-to-end script for RunPod GPU inference.

This is the FINAL script that runs on RunPod with IDM-VTON:

    1. Brand catalog auto-onboards garment data
    2. Customer clicks "Try On" → product auto-detected
    3. Customer uploads photo + enters measurements
    4. Physics pipeline preprocesses garment for selected size
    5. IDM-VTON generates the try-on image on GPU
    6. Customer toggles sizes → re-runs with different physics params

Usage on RunPod:
    python run_tryon.py \
        --product_id "cropped-borg-aviator-jacket-black" \
        --brand_id "snag_tights" \
        --customer_photo person.jpg \
        --bust 92 --waist 76 --hips 100 --height 170 \
        --size XL \
        --output output_XL.png

    # Generate all sizes at once:
    python run_tryon.py \
        --product_id "cropped-borg-aviator-jacket-black" \
        --brand_id "snag_tights" \
        --customer_photo person.jpg \
        --bust 92 --waist 76 --hips 100 --height 170 \
        --all_sizes \
        --output output_comparison.png
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import boto3
import requests
from urllib.parse import urlparse

# Add IDM-VTON, src, and gradio_demo folder to path
parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "src"))
sys.path.insert(0, str(parent_dir / "gradio_demo"))

from tryon_api import TryOnAPI
from size_aware_vton import SizeAwareVTON, classify_fit
from size_charts import compute_size_ratio, normalize_size_label
from brand_catalog import BrandCatalog

from celery import Celery
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
import os

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[CeleryIntegration()],
    traces_sample_rate=1.0,
)

app = Celery('tryon_tasks', broker='redis://localhost:6379/0')

# Global references for celery worker
tryon_pipe = None
tryon_unet_encoder = None
tryon_ip_processor = None


def load_model(model_path: str = "yisol/IDM-VTON", device: str = "cuda"):
    """Load IDM-VTON model for inference."""
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )
    
    try:
        from tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from unet_hacked_tryon import UNet2DConditionModel
        from unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    except ImportError as e:
        print(f"ERROR: IDM-VTON source code import failed: {e}")
        print(f"  sys.path includes: {[p for p in sys.path[:6]]}")
        sys.exit(1)
    
    dtype = torch.float16
    
    print("Loading IDM-VTON model...")
    print("  [1/5] VAE...")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
    
    print("  [2/5] UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=False, device_map=None).to(device)
    
    print("  [3/5] Text encoders...")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    
    print("  [4/5] Image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=dtype).to(device)
    
    print("  [5/5] UNet encoder...")
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(model_path, subfolder="unet_encoder", torch_dtype=dtype, low_cpu_mem_usage=False, device_map=None).to(device)
    
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    ip_processor = CLIPImageProcessor()
    
    pipe = TryonPipeline.from_pretrained(
        model_path, unet=unet, vae=vae,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        feature_extractor=ip_processor,
        image_encoder=image_encoder, scheduler=noise_scheduler,
        torch_dtype=dtype,
    ).to(device)
    
    print("  ✓ Model loaded!")
    return pipe, unet_encoder, ip_processor


@app.task(bind=True)
def run_inference(
    self,
    person_image_path: str,
    garment_image_path: str,
    agnostic_mask_path: str,
    positive_prompt: str,
    negative_prompt: str,
    output_path: str,
    inpainting_strength: float = 0.7,
    num_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
    device: str = "cuda",
) -> str:
    """Run a single IDM-VTON inference with physics-preprocessed inputs."""
    try:
        return _do_inference(person_image_path, garment_image_path, agnostic_mask_path, positive_prompt, negative_prompt, output_path, inpainting_strength, num_steps, guidance_scale, seed, device)
    except Exception as e:
        import sentry_sdk
        sentry_sdk.capture_exception(e)
        raise e

def _do_inference(
    person_image_path: str,
    garment_image_path: str,
    agnostic_mask_path: str,
    positive_prompt: str,
    negative_prompt: str,
    output_path: str,
    inpainting_strength: float = 0.7,
    num_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
    device: str = "cuda",
):
    global tryon_pipe, tryon_unet_encoder, tryon_ip_processor
    if 'tryon_pipe' not in globals() or tryon_pipe is None:
        tryon_pipe, tryon_unet_encoder, tryon_ip_processor = load_model("yisol/IDM-VTON", device)
        
    pipe = tryon_pipe
    unet_encoder = tryon_unet_encoder
    
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    from torchvision import transforms
    
    pipe.unet_encoder = unet_encoder
    pipe.to(device)
    unet_encoder.to(device)
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Apply background standardization before inference
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    try:
        from fitfusion.utils.preprocessing import standardize_background
        person_image_path = standardize_background(person_image_path)
    except Exception as _bg_err:
        print(f"[run_tryon] background removal skipped: {_bg_err}")

    person_image = Image.open(person_image_path).convert("RGB")
    garment_image = Image.open(garment_image_path).convert("RGB")
    agnostic_mask = Image.open(agnostic_mask_path).convert("L")
    
    # Format basic sizes
    garm_img = garment_image.resize((768, 1024))
    human_img = person_image.resize((768, 1024))
    
    mask = agnostic_mask.resize((768, 1024)).convert("L")
    mask_np = np.array(mask) > 128
    mask_out = Image.fromarray((mask_np * 255).astype(np.uint8))
    
    # DensePose Extraction on fly
    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    idm_dir = Path(__file__).resolve().parent
    config_path = str(idm_dir / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml")
    ckpt_path = str(idm_dir / "ckpt" / "densepose" / "model_final_162be9.pkl")
    
    print("\n      Extracting DensePose structure...", end="")
    args = apply_net.create_argument_parser().parse_args(('show', config_path, ckpt_path, 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img_np = args.func(args, human_img_arg)
    pose_img_np = pose_img_np[:,:,::-1]
    pose_img = Image.fromarray(pose_img_np).resize((768,1024))
    print("Done.")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                positive_prompt, do_classifier_free_guidance=True, negative_prompt=negative_prompt, num_images_per_prompt=1
            )
            
            (
                prompt_embeds_c, _, _, _,
            ) = pipe.encode_prompt(
                positive_prompt, do_classifier_free_guidance=False, negative_prompt=negative_prompt, num_images_per_prompt=1
            )
            
            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed)
            
            output = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=num_steps,
                generator=generator,
                strength=1.0, 
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask_out,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img,
                guidance_scale=guidance_scale,
            )
            
    output_image = output[0][0]
    output_image.save(output_path)
    return output_path


def generate_single_size(
    api: TryOnAPI,
    product_id: str,
    brand_id: str,
    customer_photo: str,
    bust: float,
    waist: float,
    hips: float,
    height: float,
    selected_size: str,
    skip_inference: bool=False,
    output_path: str = "output.png",
    num_steps: int = 30,
    seed: int = 42,
):
    """Generate a single size try-on image."""
    print(f"\n{'='*60}")
    print(f"  FitFusion — Size-Aware Try-On")
    print(f"  Product: {product_id} | Size: {selected_size}")
    print(f"  Customer: B={bust} W={waist} H={hips} Ht={height}")
    print(f"{'='*60}\n")
    
    # Step 1: Init (auto-detect garment)
    init = api.init_tryon(product_id, brand_id)
    if init["status"] != "ready":
        print(f"Error: {init.get('message')}")
        return None
    
    garment_info = init["garment"]
    print(f"  Garment: {garment_info['name']} ({garment_info['type']})")
    print(f"  Available sizes: {garment_info['available_sizes']}")
    
    # Step 2: Generate preprocessed inputs
    result = api.generate_tryon(
        product_id=product_id,
        brand_id=brand_id,
        customer_photo=customer_photo,
        bust_cm=bust, waist_cm=waist,
        hips_cm=hips, height_cm=height,
        selected_size=selected_size,
    )
    
    if result["status"] != "success":
        print(f"Error: {result.get('message')}")
        return None
    
    fit = result["fit_info"]
    match = result["body_match"]
    
    print(f"\n  Body match: {match['model_name']} ({match['similarity']}) ")
    print(f"  Fit type: {fit['fit_type']} — {fit['description']}")
    print(f"  Width: {fit['width_ratio']}, Feel: {fit['fabric_feel']}")
    print(f"  Preprocessing: {result['processing_time_ms']}ms")
    
    # Step 3: Run IDM-VTON inference (if model is loaded)
    if not skip_inference:
        print(f"\n  Running IDM-VTON inference ({num_steps} steps)...")
        
        session_dir = Path(result["session_dir"])
        
        # Load prompts
        with open(session_dir / "session.json") as f:
            session = json.load(f)
        
        # Strip the old prompt overrides and use the raw product description instead.
        raw_description = garment_info.get("name", product_id).lower()
        pos_prompt = f"high quality fashion photograph, {raw_description}"
        neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        strength = session["fit_profile"]["inpainting_strength"]
        
        tryon_result_path = run_inference(
            person_image_path=str(session_dir / "person_resized.png"),
            garment_image_path=str(session_dir / "garment_preprocessed.png"),
            agnostic_mask_path=str(session_dir / "mask_adapted.png"),
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt,
            output_path=output_path,
            inpainting_strength=strength,
            num_steps=num_steps,
            seed=seed,
        )
        
        print(f"\n  ✓ Try-on image saved: {tryon_result_path}")
        return tryon_result_path
    else:
        print(f"\n  ⚠ No GPU model loaded — preprocessed files saved to:")
        print(f"    {result['session_dir']}/")
        print(f"    Ready for IDM-VTON inference on GPU.")
        return None


def generate_all_sizes(
    api: TryOnAPI,
    product_id: str,
    brand_id: str,
    customer_photo: str,
    bust: float, waist: float, hips: float, height: float,
    skip_inference: bool=False,
    output_dir: str = "output_sizes",
    num_steps: int = 30,
    seed: int = 42,
):
    """Generate try-on images for all available sizes side by side."""
    init = api.init_tryon(product_id, brand_id)
    if init["status"] != "ready":
        print(f"Error: {init.get('message')}")
        return
    
    sizes = init["garment"]["available_sizes"]
    normalized_sizes = [normalize_size_label(s) for s in sizes]
    
    print(f"\nGenerating {len(sizes)} sizes: {sizes}")
    print(f"Normalized: {normalized_sizes}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for size in sizes:
        norm_size = normalize_size_label(size)
        out_path = os.path.join(output_dir, f"tryon_{norm_size}.png")
        
        result = generate_single_size(
            api, product_id, brand_id, customer_photo,
            bust, waist, hips, height, size,
            skip_inference,
            output_path=out_path, num_steps=num_steps, seed=seed,
        )
        results.append((size, norm_size, out_path, result))
    
    # Create side-by-side comparison
    print(f"\n{'='*60}")
    print("  Size Comparison Summary")
    print(f"{'='*60}")
    
    for size, norm, path, _ in results:
        fit = classify_fit(init["garment"]["type"], size, "M")
        print(f"  {norm:>4s} ({size:>3s}): {fit.fit_type.name:<12s} "
              f"W={fit.width_ratio:.2f}x  Warp={fit.warp_intensity:.1f}  "
              f"Tension={fit.fabric_tension:.2f}")
    
    print(f"\n  Output directory: {output_dir}/")


def download_customer_asset(s3_image_url: str, local_destination: str):
    """Securely downloads the image to the local destination before the IDM-VTON pipeline initializes."""
    os.makedirs(os.path.dirname(local_destination), exist_ok=True)
    
    if s3_image_url.startswith("file://"):
        # Local file path — just copy it
        import shutil
        local_src = s3_image_url[len("file://"):]
        shutil.copy2(local_src, local_destination)
    elif s3_image_url.startswith("http"):
        response = requests.get(s3_image_url, stream=True)
        response.raise_for_status()
        with open(local_destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    elif s3_image_url.startswith("s3://"):
        parsed = urlparse(s3_image_url)
        s3 = boto3.client('s3')
        s3.download_file(parsed.netloc, parsed.path.lstrip('/'), local_destination)
    else:
        raise ValueError("Invalid s3_image_url format: Must start with file://, http, or s3://")


def main():
    parser = argparse.ArgumentParser(
        description="FitFusion — Size-Aware Virtual Try-On",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single size:
  python run_tryon.py --product_id "jacket-id" --brand_id "snag_tights" \\
    --customer_photo person.jpg --bust 92 --waist 76 --hips 100 --size XL

  # All sizes comparison:
  python run_tryon.py --product_id "jacket-id" --brand_id "snag_tights" \\
    --customer_photo person.jpg --bust 92 --waist 76 --hips 100 --all_sizes

  # Preprocessing only (no GPU needed):
  python run_tryon.py --product_id "jacket-id" --brand_id "snag_tights" \\
    --customer_photo person.jpg --bust 92 --waist 76 --hips 100 --size XL --no_model
        """
    )
    
    parser.add_argument("--product_id", required=True, help="Product ID from brand catalog")
    parser.add_argument("--brand_id", required=True, help="Brand ID")
    parser.add_argument("--s3_image_url", required=True, help="S3 URL or presigned link for customer photo")
    parser.add_argument("--bust", type=float, required=True, help="Customer bust (cm)")
    parser.add_argument("--waist", type=float, required=True, help="Customer waist (cm)")
    parser.add_argument("--hips", type=float, required=True, help="Customer hips (cm)")
    parser.add_argument("--height", type=float, default=170, help="Customer height (cm)")
    parser.add_argument("--size", default="M", help="Garment size to try (S/M/L/XL/etc)")
    parser.add_argument("--all_sizes", action="store_true", help="Generate all available sizes")
    parser.add_argument("--output", default="output_tryon.png", help="Output image path")
    parser.add_argument("--model_path", default="yisol/IDM-VTON", help="IDM-VTON model path")
    # Default catalog_dir is relative to the project root (one level above IDM-VTON/)
    _project_root = str(Path(__file__).resolve().parent.parent)
    _default_catalog = os.path.join(_project_root, "data", "brand_catalog")
    parser.add_argument("--catalog_dir", default=_default_catalog, help="Brand catalog dir")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_model", action="store_true", help="Skip model loading (preprocess only)")
    
    args = parser.parse_args()
    
    # Download S3 Asset Pipeline Layer
    local_customer_photo = "/workspace/FitFusion/data/customer_photo_downloaded.jpg"
    download_customer_asset(args.s3_image_url, local_customer_photo)
    
    # Initialize API
    api = TryOnAPI(catalog_dir=args.catalog_dir)
    
    if not api.catalog.garments:
        print(f"ERROR: Catalog is empty. Check catalog_dir: {args.catalog_dir}")
        print(f"  catalog.json should be at: {os.path.join(args.catalog_dir, 'catalog.json')}")
        sys.exit(1)
    
    # Load model (unless --no_model)
    global tryon_pipe, tryon_unet_encoder, tryon_ip_processor
    if not args.no_model:
        try:
            tryon_pipe, tryon_unet_encoder, tryon_ip_processor = load_model(args.model_path)
        except Exception as e:
            import traceback
            print(f"Could not load model:")
            traceback.print_exc()
            print("Continuing with preprocessing only.\n")
            args.no_model = True
    
    if args.all_sizes:
        generate_all_sizes(
            api, args.product_id, args.brand_id, local_customer_photo,
            args.bust, args.waist, args.hips, args.height,
            skip_inference=args.no_model,
            output_dir=os.path.dirname(args.output) or "output_sizes",
            num_steps=args.steps, seed=args.seed,
        )
    else:
        generate_single_size(
            api, args.product_id, args.brand_id, local_customer_photo,
            args.bust, args.waist, args.hips, args.height, args.size,
            skip_inference=args.no_model,
            output_path=args.output, num_steps=args.steps, seed=args.seed,
        )


if __name__ == "__main__":
    main()
