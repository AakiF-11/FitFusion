"""
FitFusion — Production Entry Point
====================================
Size-Aware Virtual Try-On Pipeline

Execution:
    python run_pipeline.py \
        --person_image  inputs/person/customer.jpg \
        --garment_image inputs/garment/jacket.jpg  \
        --target_size   XL                         \
        --ckpt_dir      ./ckpt                     \
        --output_dir    ./outputs

All paths are resolved at runtime. Zero hardcoded values.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fitfusion")


# ── Argument Parser ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="FitFusion: Size-Aware Virtual Try-On Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required inputs ────────────────────────────────────────────────────────
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--person_image",
        type=Path,
        required=True,
        help="Path to the customer photo (JPG/PNG). Must be a full-body shot.",
    )
    io.add_argument(
        "--garment_image",
        type=Path,
        required=True,
        help="Path to the flat garment product photo (JPG/PNG).",
    )
    io.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory where result images and metadata are saved.",
    )

    # ── Size configuration ─────────────────────────────────────────────────────
    sizing = parser.add_argument_group("Sizing")
    sizing.add_argument(
        "--target_size",
        type=str,
        required=True,
        choices=["4XS","3XS","2XS","XS","S","M","L","XL","2XL","3XL","4XL"],
        help="The garment size to simulate wearing (e.g. XL).",
    )
    sizing.add_argument(
        "--person_size",
        type=str,
        default="M",
        choices=["4XS","3XS","2XS","XS","S","M","L","XL","2XL","3XL","4XL"],
        help="The customer's actual body size baseline.",
    )
    sizing.add_argument(
        "--garment_type",
        type=str,
        default="top",
        choices=["top", "pants", "dress", "skirt", "jacket", "outerwear"],
        help="Category of the garment being tried on.",
    )

    # ── Model weights ──────────────────────────────────────────────────────────
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--ckpt_dir",
        type=Path,
        default=Path("./ckpt"),
        help="Path to the IDM-VTON model weights directory.",
    )

    # ── Inference settings ─────────────────────────────────────────────────────
    inference = parser.add_argument_group("Inference")
    inference.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion inference steps. Higher = better quality, slower.",
    )
    inference.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale.",
    )
    inference.add_argument(
        "--width",
        type=int,
        default=768,
        help="Inference image width (must match IDM-VTON training resolution).",
    )
    inference.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Inference image height (must match IDM-VTON training resolution).",
    )
    inference.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    inference.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device.",
    )

    # ── Pipeline behaviour ─────────────────────────────────────────────────────
    pipeline = parser.add_argument_group("Pipeline")
    pipeline.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip background standardization (use if image is already clean).",
    )
    pipeline.add_argument(
        "--skip_skin_restore",
        action="store_true",
        help="Skip post-process skin compositing step.",
    )
    pipeline.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save intermediate images (masked, warped garment, densepose).",
    )

    return parser


# ── Path validation ────────────────────────────────────────────────────────────
def validate_args(args: argparse.Namespace) -> None:
    errors = []

    if not args.person_image.exists():
        errors.append(f"--person_image not found: {args.person_image}")
    if not args.garment_image.exists():
        errors.append(f"--garment_image not found: {args.garment_image}")
    if not args.ckpt_dir.exists():
        errors.append(f"--ckpt_dir not found: {args.ckpt_dir}")

    required_ckpt_subdirs = [
        "unet", "vae", "text_encoder", "text_encoder_2",
        "tokenizer", "tokenizer_2", "image_encoder", "unet_encoder", "scheduler",
    ]
    for sub in required_ckpt_subdirs:
        if not (args.ckpt_dir / sub).exists():
            errors.append(f"Missing checkpoint subfolder: {args.ckpt_dir / sub}")

    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)


# ── Output directory setup ─────────────────────────────────────────────────────
def prepare_output_dir(output_dir: Path, person_name: str) -> Path:
    """Creates a unique job directory: outputs/<person_stem>_<timestamp>/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = output_dir / f"{person_name}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    if not (job_dir / "intermediates").exists():
        (job_dir / "intermediates").mkdir()
    return job_dir


# ── Pipeline stages ────────────────────────────────────────────────────────────
def stage_preprocess(args: argparse.Namespace, job_dir: Path):
    """
    Stage 1: Standardize inputs.
    - Strip background from customer photo → solid studio gray
    - Validate image dimensions
    """
    log.info("[Stage 1/6] Preprocessing inputs...")

    from src.preprocessing.background import standardize_background
    from src.preprocessing.validator import validate_image

    person_path = str(args.person_image)
    garment_path = str(args.garment_image)

    if not args.skip_preprocessing:
        person_path = standardize_background(person_path, save_dir=str(job_dir / "intermediates"))
        log.info(f"  Background standardized → {person_path}")

    validate_image(person_path, min_height=512, min_width=384)
    validate_image(garment_path, min_height=512, min_width=384)

    return person_path, garment_path


def stage_pose_estimation(person_path: str, args: argparse.Namespace, job_dir: Path):
    """
    Stage 2: Extract pose from customer photo.
    - OpenPose keypoints (body skeleton)
    - DensePose UV map (surface normals for UV-guided inpainting)
    """
    log.info("[Stage 2/6] Estimating pose (OpenPose + DensePose)...")

    from src.pose.openpose import extract_openpose_keypoints
    from src.pose.densepose import generate_densepose_map

    openpose_kpts = extract_openpose_keypoints(
        image_path=person_path,
        ckpt_dir=str(args.ckpt_dir / "openpose"),
    )
    densepose_map = generate_densepose_map(
        image_path=person_path,
        ckpt_dir=str(args.ckpt_dir / "densepose"),
        output_dir=str(job_dir / "intermediates") if args.save_intermediates else None,
    )

    return openpose_kpts, densepose_map


def stage_masking(person_path: str, openpose_kpts, args: argparse.Namespace, job_dir: Path):
    """
    Stage 3: Generate agnostic mask.
    - SCHP human parsing segmentation
    - Confidence scoring (reject impossible arm geometries)
    - Size-adaptive mask scaling based on target_size vs person_size
    """
    log.info("[Stage 3/6] Generating size-adaptive agnostic mask...")

    from src.masking.human_parsing import run_human_parsing
    from src.masking.confidence import score_mask_validity
    from src.masking.adaptive_mask import generate_adaptive_mask

    schp_mask = run_human_parsing(
        image_path=person_path,
        ckpt_dir=str(args.ckpt_dir / "humanparsing"),
    )

    confidence, valid, reason = score_mask_validity(schp_mask, openpose_kpts)
    if not valid:
        log.error(f"  Mask confidence {confidence:.2f} — halting: {reason}")
        sys.exit(1)
    log.info(f"  Mask confidence: {confidence:.2f} ✓")

    agnostic_mask = generate_adaptive_mask(
        schp_mask=schp_mask,
        garment_type=args.garment_type,
        person_size=args.person_size,
        target_size=args.target_size,
        output_dir=str(job_dir / "intermediates") if args.save_intermediates else None,
    )

    return schp_mask, agnostic_mask


def stage_size_physics(garment_path: str, args: argparse.Namespace, job_dir: Path):
    """
    Stage 4: Apply size physics to garment.
    - Compute width/length ratios from size charts
    - Resize flat garment image by those ratios
    - TPS warp toward body contour
    """
    log.info("[Stage 4/6] Applying size physics to garment...")

    from src.size_physics.physics import SizeAwareVTON
    from src.size_physics.resizer import GarmentResizer

    resizer = GarmentResizer(target_resolution=(args.width, args.height))
    physics = SizeAwareVTON()

    params = physics.compute_physics_params(
        person_size=args.person_size,
        garment_size=args.target_size,
        garment_type=args.garment_type,
    )
    log.info(f"  Fit class: {params.fit_type.name}  |  Width ratio: {params.width_ratio:.3f}")

    warped_garment = resizer.resize_garment(
        garment_image_path=garment_path,
        garment_type=args.garment_type,
        person_size=args.person_size,
        garment_size=args.target_size,
        save_dir=str(job_dir / "intermediates") if args.save_intermediates else None,
    )

    return warped_garment, params


def stage_inference(
    person_path: str,
    warped_garment,
    agnostic_mask,
    densepose_map,
    physics_params,
    args: argparse.Namespace,
    job_dir: Path,
):
    """
    Stage 5: Run IDM-VTON diffusion inference.
    - Load model weights from --ckpt_dir
    - Generate try-on image
    """
    log.info("[Stage 5/6] Running IDM-VTON inference on GPU...")

    from src.inference.model_loader import load_idm_vton_pipeline
    from src.inference.tryon import run_tryon

    pipeline = load_idm_vton_pipeline(
        ckpt_dir=str(args.ckpt_dir),
        device=args.device,
    )

    result_image = run_tryon(
        pipeline=pipeline,
        person_image_path=person_path,
        warped_garment=warped_garment,
        agnostic_mask=agnostic_mask,
        densepose_map=densepose_map,
        physics_params=physics_params,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    return result_image


def stage_postprocess(result_image, person_path: str, schp_mask, args: argparse.Namespace, job_dir: Path):
    """
    Stage 6: Post-process output.
    - Restore original skin/tattoos over generated pixels
    - Save final result + metadata
    """
    log.info("[Stage 6/6] Post-processing output...")

    from PIL import Image
    import json

    if not args.skip_skin_restore:
        from src.masking.compositor import restore_original_skin
        original = Image.open(person_path).convert("RGB")
        result_image = restore_original_skin(original, result_image, schp_mask)
        log.info("  Skin compositing applied.")

    output_path = job_dir / "result.png"
    result_image.save(str(output_path))
    log.info(f"  Result saved → {output_path}")

    metadata = {
        "person_image": str(args.person_image),
        "garment_image": str(args.garment_image),
        "target_size": args.target_size,
        "person_size": args.person_size,
        "garment_type": args.garment_type,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    with open(job_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("FitFusion — Size-Aware Virtual Try-On")
    log.info("=" * 60)
    log.info(f"  Person image  : {args.person_image}")
    log.info(f"  Garment image : {args.garment_image}")
    log.info(f"  Target size   : {args.target_size}")
    log.info(f"  Person size   : {args.person_size}")
    log.info(f"  Garment type  : {args.garment_type}")
    log.info(f"  Checkpoint dir: {args.ckpt_dir}")
    log.info(f"  Output dir    : {args.output_dir}")
    log.info("=" * 60)

    validate_args(args)

    job_dir = prepare_output_dir(args.output_dir, args.person_image.stem)
    log.info(f"Job directory: {job_dir}")

    person_path, garment_path   = stage_preprocess(args, job_dir)
    openpose_kpts, densepose_map = stage_pose_estimation(person_path, args, job_dir)
    schp_mask, agnostic_mask    = stage_masking(person_path, openpose_kpts, args, job_dir)
    warped_garment, params      = stage_size_physics(garment_path, args, job_dir)
    result_image                = stage_inference(person_path, warped_garment, agnostic_mask, densepose_map, params, args, job_dir)
    output_path                 = stage_postprocess(result_image, person_path, schp_mask, args, job_dir)

    log.info("=" * 60)
    log.info(f"Pipeline complete → {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
