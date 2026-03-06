"""
FitFusion — Size-Aware IDM-VTON Training with QLoRA + MeasurementEncoder
=========================================================================
Modified from train_xl.py. Key changes:
  1. QLoRA wrapping of UNet (LoRA rank=16, alpha=32, target=[to_q, to_k, to_v, to_out.0])
  2. MeasurementEncoder: 6-measurement MLP → conditioning tokens injected into cross-attention
  3. SizeAwareVitonHDDataset: loads measurements.json alongside VITON-HD images
  4. Checkpoint saving: LoRA adapters + MeasurementEncoder + Resampler (not full pipeline)
  5. 4-bit quantization of UNet base weights via BitsAndBytes
  6. Resume from LoRA checkpoint support

Measurement injection path:
  measurement_tokens (B,4,2048) + ip_tokens (B,16,2048) → image_embeds (B,20,2048)
  → UNet concatenates to encoder_hidden_states: [text(77) | measurement(4) | ip(16)]
  → IPAttnProcessor2_0 splits at len-16: text+measurement get normal xattn, ip get ip_xattn
"""

import os
import random
import argparse
import json
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
)

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple, List, Optional, Dict
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF

# QLoRA / PEFT imports
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb

# MeasurementEncoder
from measurement_encoder import (
    MeasurementEncoder,
    normalize_measurements,
    GARMENT_TYPE_MAP,
    SIZE_TO_INDEX,
)


# ─── Dataset ────────────────────────────────────────────────────────────────

class SizeAwareVitonHDDataset(data.Dataset):
    """
    Extended VITON-HD dataset that also loads body measurements.

    Expected directory structure:
        dataroot/
            train/
                image/          ← person images
                cloth/          ← garment images
                agnostic-mask/  ← *_mask.png
                image-densepose/← DensePose maps
            train_pairs.txt     ← "image_name cloth_name" per line
            vitonhd_train_tagged.json   ← garment annotations
            measurements.json   ← body measurements per image

    measurements.json format:
        {
            "image_name.jpg": {
                "bust_cm": 88.0,
                "waist_cm": 68.0,
                "hips_cm": 96.0,
                "height_cm": 170.0,
                "size_label": "M",
                "garment_type": "top"
            }
        }
    """

    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size

        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()

        # ── Garment annotations ──
        tagged_path = os.path.join(
            dataroot_path, phase, f"vitonhd_{phase}_tagged.json"
        )
        self.annotation_pair = {}
        if os.path.exists(tagged_path):
            with open(tagged_path) as f:
                data1 = json.load(f)
            annotation_list = ["sleeveLength", "neckLine", "item"]
            for k, v in data1.items():
                for elem in v:
                    annotation_str = ""
                    for template in annotation_list:
                        for tag in elem["tag_info"]:
                            if (
                                tag["tag_name"] == template
                                and tag["tag_category"] is not None
                            ):
                                annotation_str += tag["tag_category"] + " "
                    self.annotation_pair[elem["file_name"]] = annotation_str.strip()

        # ── Body measurements ──
        measurements_path = os.path.join(dataroot_path, "measurements.json")
        self.measurements = {}
        if os.path.exists(measurements_path):
            with open(measurements_path) as f:
                self.measurements = json.load(f)
        else:
            print(f"[WARN] No measurements.json found at {measurements_path}")
            print("       Training without measurement conditioning.")

        # ── Pairs ──
        self.order = order
        im_names, c_names, dataroot_names = [], [], []

        pairs_file = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    im_name, c_name = parts[0], parts[1]
                else:
                    im_name = parts[0]
                    c_name = im_name
                if phase == "train":
                    c_name = im_name  # paired training: cloth = own garment
                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.clip_processor = CLIPImageProcessor()

        print(f"[Dataset] Loaded {len(self.im_names)} pairs from {phase}")
        print(f"[Dataset] Measurements available for {len(self.measurements)} images")

    def __len__(self):
        return len(self.im_names)

    def _get_measurement_vector(self, im_name: str, c_name: str) -> (torch.Tensor, str):
        """Return normalized measurement vector for an image and fit class string, or zeros if unavailable."""
        if im_name in self.measurements:
            m = self.measurements[im_name]
            target_size_label = None
            if c_name in self.measurements:
                target_size_label = self.measurements[c_name].get("size_label", "M")

            # Calculate Fit Class based on sizes
            fit_class = ""
            if target_size_label:
                size_idx = SIZE_TO_INDEX.get(m.get("size_label", "M").upper(), 5)
                target_idx = SIZE_TO_INDEX.get(target_size_label.upper(), 5)
                delta = target_idx - size_idx
                if delta <= -2: fit_class = "[FF_BURSTING]"
                elif delta == -1: fit_class = "[FF_TIGHT]"
                elif delta == 0: fit_class = "[FF_PERFECT]"
                elif delta == 1: fit_class = "[FF_RELAXED]"
                elif delta == 2: fit_class = "[FF_BAGGY]"
                else: fit_class = "[FF_TENT]"

            # Determine garment size label from the cloth entry
            garment_size_label = None
            if c_name in self.measurements:
                garment_size_label = self.measurements[c_name].get("size_label", None)

            vec = normalize_measurements(
                bust_cm=m.get("bust_cm", 0),
                waist_cm=m.get("waist_cm", 0),
                hips_cm=m.get("hips_cm", 0),
                height_cm=m.get("height_cm", 0),
                size_label=m.get("size_label", "M"),
                garment_type=m.get("garment_type", "top"),
                target_size_label=target_size_label,
                garment_size_label=garment_size_label,
            )
            return vec, fit_class
        else:
            # Return zeros (MeasurementEncoder with zero input → near-zero output due to init)
            return torch.zeros(12, dtype=torch.float32), ""

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        cloth_annotation = self.annotation_pair.get(c_name, "clothing")

        cloth = Image.open(
            os.path.join(self.dataroot, self.phase, "cloth", c_name)
        )
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width, self.height))

        image = self.transform(im_pil_big)

        # Agnostic mask
        # Derive mask name: 000001.jpg → 000001_mask.png
        mask_stem = os.path.splitext(im_name)[0]
        mask_name = mask_stem + "_mask.png"
        mask = Image.open(
            os.path.join(self.dataroot, self.phase, "agnostic-mask", mask_name)
        ).resize((self.width, self.height))
        mask = self.toTensor(mask)[:1]

        # DensePose
        densepose_map = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        )
        pose_img = self.toTensor(densepose_map)

        # ── Data augmentation (train only) ──
        if self.phase == "train":
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)

            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5
                )
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue,
                )
                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)
                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                image = transforms.functional.affine(
                    image, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                mask = transforms.functional.affine(
                    mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                pose_img = transforms.functional.affine(
                    pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )

            if random.random() > 0.5:
                shift_x = random.uniform(-0.2, 0.2)
                shift_y = random.uniform(-0.2, 0.2)
                image = transforms.functional.affine(
                    image,
                    angle=0,
                    translate=[shift_x * image.shape[-1], shift_y * image.shape[-2]],
                    scale=1,
                    shear=0,
                )
                mask = transforms.functional.affine(
                    mask,
                    angle=0,
                    translate=[shift_x * mask.shape[-1], shift_y * mask.shape[-2]],
                    scale=1,
                    shear=0,
                )
                pose_img = transforms.functional.affine(
                    pose_img,
                    angle=0,
                    translate=[
                        shift_x * pose_img.shape[-1],
                        shift_y * pose_img.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )

        # ── Finalize ──
        mask = 1 - mask
        cloth_trim = self.clip_processor(images=cloth, return_tensors="pt").pixel_values
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        im_mask = image * mask
        pose_img = self.norm(pose_img)

        # ── Measurements ──
        measurement_vec, fit_class = self._get_measurement_vector(im_name, c_name)

        caption = "model is wearing " + cloth_annotation
        if fit_class:
            caption += f", {fit_class} fit"

        result = {
            "c_name": c_name,
            "image": image,
            "cloth": cloth_trim,
            "cloth_pure": self.transform(cloth),
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption": caption,
            "caption_cloth": "a photo of " + cloth_annotation,
            "annotation": cloth_annotation,
            "pose_img": pose_img,
            "measurements": measurement_vec,  # (12,)
        }
        return result


# ─── Argument Parser ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FitFusion: Size-aware IDM-VTON training with QLoRA"
    )

    # Model paths
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    )
    parser.add_argument(
        "--pretrained_garmentnet_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="ckpt/image_encoder",
    )

    # QLoRA config
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Quantize UNet base weights to 4-bit NF4",
    )

    # MeasurementEncoder config
    parser.add_argument(
        "--num_measurement_tokens",
        type=int,
        default=4,
        help="Number of measurement conditioning tokens",
    )
    parser.add_argument(
        "--measurement_hidden_dim",
        type=int,
        default=512,
        help="Hidden dim in MeasurementEncoder MLP",
    )

    # Training config
    parser.add_argument("--checkpointing_epoch", type=int, default=10)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="output_qlora")
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--num_tokens", type=int, default=16, help="IP adapter tokens")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Higher LR for LoRA")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Smaller batch for QLoRA")
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--noise_offset", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_dir", type=str, default="data/fitfusion_vitonhd")

    # Resume
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir with lora_adapter/, measurement_encoder.pt, resampler.pt",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


# ─── Main Training Loop ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Load base models (frozen)
    # ═══════════════════════════════════════════════════════════════════════
    print("[1/8] Loading base models...")

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        rescale_betas_zero_snr=True,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_garmentnet_path, subfolder="unet_encoder", use_safetensors=True, torch_dtype=torch.float16
    )
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_path
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Load & customize UNet (then wrap with LoRA)
    # ═══════════════════════════════════════════════════════════════════════
    print("[2/8] Loading UNet and applying LoRA...")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float16
    )

    # Set up IP-adapter integration
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"

    # Load IP-adapter attention processor weights
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    # IP-adapter Resampler (projects CLIP image features → cross-attention tokens)
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(accelerator.device, dtype=torch.float32)
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    image_proj_model.requires_grad_(True)  # Trainable

    unet.encoder_hid_proj = image_proj_model

    # Replace conv_in with 13-channel version (4 noisy + 4 masked + 1 mask + 4 pose)
    if unet.config.in_channels != 13:
        conv_new = torch.nn.Conv2d(
            in_channels=4 + 4 + 1 + 4,
            out_channels=unet.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )
        torch.nn.init.kaiming_normal_(conv_new.weight)
        conv_new.weight.data = conv_new.weight.data * 0.0
        # Check if the base unet has 9 or 4 channels depending on checkpoint source
        original_channels = unet.conv_in.weight.data.shape[1]
        conv_new.weight.data[:, :original_channels] = unet.conv_in.weight.data
        conv_new.bias.data = unet.conv_in.bias.data
        unet.conv_in = conv_new
        unet.config["in_channels"] = 13
        unet.config.in_channels = 13

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
    )

    # Freeze base UNet first, then add LoRA
    unet.requires_grad_(False)
    unet = get_peft_model(unet, lora_config)

    # CRITICAL: Unfreeze conv_in so new pose channels (9-12) can learn
    # unet.requires_grad_(False) froze it, but conv_in is NOT a LoRA target
    for param in unet.base_model.model.conv_in.parameters():
        param.requires_grad = True

    # CRITICAL: Unfreeze Resampler (image_proj_model / encoder_hid_proj)
    # It was attached as a UNet submodule before freezing, so it got frozen too
    image_proj_model.requires_grad_(True)

    # Print trainable parameter summary
    unet.print_trainable_parameters()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Create MeasurementEncoder
    # ═══════════════════════════════════════════════════════════════════════
    print("[3/8] Creating MeasurementEncoder...")

    measurement_encoder = MeasurementEncoder(
        num_measurements=12,
        output_dim=unet.config.cross_attention_dim,
        num_tokens=args.num_measurement_tokens,
        hidden_dim=args.measurement_hidden_dim,
    ).to(accelerator.device, dtype=torch.float32)
    measurement_encoder.requires_grad_(True)

    total_me_params = sum(p.numel() for p in measurement_encoder.parameters())
    print(f"  MeasurementEncoder params: {total_me_params:,}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Move to devices & freeze frozen modules
    # ═══════════════════════════════════════════════════════════════════════
    print("[4/8] Setting up devices and precision...")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_encoder.enable_gradient_checkpointing()

    unet.train()
    measurement_encoder.train()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Optimizer — only trainable params
    # ═══════════════════════════════════════════════════════════════════════
    print("[5/8] Setting up optimizer...")

    if args.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Trainable: LoRA params + MeasurementEncoder + Resampler (image_proj_model)
    trainable_params = list(
        filter(lambda p: p.requires_grad, unet.parameters())
    ) + list(measurement_encoder.parameters()) + list(image_proj_model.parameters())

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Total trainable parameters: {total_trainable:,}")

    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Datasets
    # ═══════════════════════════════════════════════════════════════════════
    print("[6/8] Loading datasets...")

    train_dataset = SizeAwareVitonHDDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="paired",
        size=(args.height, args.width),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=4,
    )
    test_dataset = SizeAwareVitonHDDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="paired",
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=2,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Accelerate prepare
    # ═══════════════════════════════════════════════════════════════════════
    (
        unet,
        measurement_encoder,
        image_proj_model,
        unet_encoder,
        image_encoder,
        optimizer,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        unet,
        measurement_encoder,
        image_proj_model,
        unet_encoder,
        image_encoder,
        optimizer,
        train_dataloader,
        test_dataloader,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    # ── Resume from checkpoint ──
    initial_global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        print(f"  Resuming from {ckpt_path}")

        # Load LoRA adapter
        lora_path = os.path.join(ckpt_path, "lora_adapter")
        if os.path.exists(lora_path):
            unwrapped = accelerator.unwrap_model(unet)
            unwrapped.load_adapter(lora_path, "default")
            print("  ✓ LoRA adapter loaded")

        # Load MeasurementEncoder
        me_path = os.path.join(ckpt_path, "measurement_encoder.pt")
        if os.path.exists(me_path):
            me_state = torch.load(me_path, map_location="cpu")
            accelerator.unwrap_model(measurement_encoder).load_state_dict(me_state)
            print("  ✓ MeasurementEncoder loaded")

        # Load Resampler
        res_path = os.path.join(ckpt_path, "resampler.pt")
        if os.path.exists(res_path):
            res_state = torch.load(res_path, map_location="cpu")
            accelerator.unwrap_model(image_proj_model).load_state_dict(res_state)
            print("  ✓ Resampler loaded")

        # Load optimizer state
        opt_path = os.path.join(ckpt_path, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu")
            optimizer.load_state_dict(opt_state)
            print("  ✓ Optimizer state loaded")

        # Load training state
        state_path = os.path.join(ckpt_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                tstate = json.load(f)
            initial_global_step = tstate.get("global_step", 0)
            first_epoch = tstate.get("epoch", 0)
            print(f"  ✓ Resuming from step {initial_global_step}, epoch {first_epoch}")

    # ═══════════════════════════════════════════════════════════════════════
    # 8. Training loop
    # ═══════════════════════════════════════════════════════════════════════
    print("[7/8] Starting training...")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Steps per epoch: {num_update_steps_per_epoch}")
    print(f"  Total steps: {args.max_train_steps}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  Measurement tokens: {args.num_measurement_tokens}")
    print()

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = initial_global_step
    train_loss = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        
        # ── Two-Phased Training Protocol ──
        if epoch < 5:
            # Phase 1: Freeze UNet LoRA and IP-Adapter, train only MeasurementEncoder
            if accelerator.is_main_process:
                print(f"[Epoch {epoch}] Phase 1: Freezing UNet and IP-Adapter. Training info only.")
            unet.requires_grad_(False)
            image_proj_model.requires_grad_(False)
            measurement_encoder.requires_grad_(True)
        else:
            # Phase 2: End-to-end fine-tuning
            if accelerator.is_main_process:
                print(f"[Epoch {epoch}] Phase 2: Unfreezing UNet and IP-Adapter.")
            for name, param in unet.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            for param in unet.base_model.model.conv_in.parameters():
                param.requires_grad = True
            image_proj_model.requires_grad_(True)
            measurement_encoder.requires_grad_(True)

        # Calculate how many steps to skip in the first resumed epoch
        steps_to_skip = 0
        if epoch == first_epoch and initial_global_step > 0:
            steps_to_skip = initial_global_step - (first_epoch * num_update_steps_per_epoch)
            steps_to_skip = max(0, steps_to_skip * args.gradient_accumulation_steps)

        for step, batch in enumerate(train_dataloader):
            # Skip already-completed steps when resuming
            if step < steps_to_skip:
                if step % 100 == 0:
                    progress_bar.set_description(f"Skipping {step}/{steps_to_skip}")
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(measurement_encoder):

                # ── Periodic validation ──
                if global_step % args.logging_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        _run_validation(
                            args,
                            accelerator,
                            unet,
                            vae,
                            noise_scheduler,
                            tokenizer,
                            tokenizer_2,
                            text_encoder,
                            text_encoder_2,
                            image_encoder,
                            unet_encoder,
                            test_dataloader,
                            measurement_encoder,
                            image_proj_model,
                            global_step,
                        )

                # ════════════════════════════════════════════════════════════
                # Forward pass
                # ════════════════════════════════════════════════════════════

                # Encode images to latent space
                pixel_values = batch["image"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["im_mask"].reshape(batch["image"].shape).to(dtype=vae.dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["inpaint_mask"]
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(
                            masks, size=(args.height // 8, args.width // 8)
                        )
                    ]
                )
                mask = mask.reshape(-1, 1, args.height // 8, args.width // 8)

                pose_map = vae.encode(
                    batch["pose_img"].to(dtype=vae.dtype)
                ).latent_dist.sample()
                pose_map = pose_map * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)

                # 13-channel input: [noisy(4) | mask(1) | masked_latents(4) | pose(4)]
                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_latents, pose_map], dim=1
                )

                # ── Text encoding ──
                text_input_ids = tokenizer(
                    batch["caption"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch["caption"],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                encoder_output = text_encoder(
                    text_input_ids.to(accelerator.device), output_hidden_states=True
                )
                text_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(
                    text_input_ids_2.to(accelerator.device), output_hidden_states=True
                )
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                encoder_hidden_states = torch.concat(
                    [text_embeds, text_embeds_2], dim=-1
                )  # (B, 77, 2048)

                # ── Time conditioning ──
                def compute_time_ids(original_size, crops_coords_top_left=(0, 0)):
                    target_size = (args.height, args.width)
                    add_time_ids = list(
                        original_size + crops_coords_top_left + target_size
                    )
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids((args.height, args.width)) for _ in range(bsz)]
                )

                # ── IP-adapter: CLIP image → Resampler → ip_tokens ──
                img_emb_list = []
                for i in range(bsz):
                    img_emb_list.append(batch["cloth"][i])
                image_embeds = torch.cat(img_emb_list, dim=0)
                image_embeds = image_encoder(
                    image_embeds, output_hidden_states=True
                ).hidden_states[-2]
                ip_tokens = image_proj_model(image_embeds)  # (B, 16, 2048)

                # ── Measurement conditioning ──
                measurements = batch["measurements"].to(
                    accelerator.device, dtype=torch.float32
                )  # (B, 6)
                measurement_tokens = measurement_encoder(measurements)
                # Cast to same dtype as ip_tokens
                measurement_tokens = measurement_tokens.to(dtype=ip_tokens.dtype)
                # (B, num_measurement_tokens, 2048)

                # Combine: [measurement_tokens | ip_tokens]
                # In UNet forward, these get concatenated to encoder_hidden_states:
                #   [text(77) | measurement(4) | ip(16)] = 97 tokens
                # IPAttnProcessor2_0 splits at len-16: measurement goes into text xattn
                combined_image_embeds = torch.cat(
                    [measurement_tokens, ip_tokens], dim=1
                )  # (B, 20, 2048)

                unet_added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                    "image_embeds": combined_image_embeds,
                }

                # ── Garment encoder ──
                cloth_values = batch["cloth_pure"].to(
                    accelerator.device, dtype=vae.dtype
                )
                cloth_values = vae.encode(cloth_values).latent_dist.sample()
                cloth_values = cloth_values * vae.config.scaling_factor

                text_input_ids = tokenizer(
                    batch["caption_cloth"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch["caption_cloth"],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                encoder_output = text_encoder(
                    text_input_ids.to(accelerator.device), output_hidden_states=True
                )
                text_embeds_cloth = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(
                    text_input_ids_2.to(accelerator.device), output_hidden_states=True
                )
                text_embeds_2_cloth = encoder_output_2.hidden_states[-2]
                text_embeds_cloth = torch.concat(
                    [text_embeds_cloth, text_embeds_2_cloth], dim=-1
                )

                down, reference_features = unet_encoder(
                    cloth_values, timesteps, text_embeds_cloth, return_dict=False
                )
                reference_features = list(reference_features)

                # ── UNet forward ──
                noise_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    garment_features=reference_features,
                ).sample

                # ── Loss ──
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        model_input, noise, timesteps
                    )
                elif noise_scheduler.config.prediction_type == "sample":
                    target = model_input
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        noise_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        noise_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # ── Backprop ──
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_params = (
                        list(filter(lambda p: p.requires_grad, unet.parameters()))
                        + list(measurement_encoder.parameters())
                        + list(image_proj_model.parameters())
                    )
                    accelerator.clip_grad_norm_(all_params, 1.0)

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

            if accelerator.sync_gradients:
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "epoch": epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # ── Checkpoint at epoch boundary ──
        if (epoch + 1) % args.checkpointing_epoch == 0:
            if accelerator.is_main_process:
                _save_checkpoint(
                    args, accelerator, unet, measurement_encoder,
                    image_proj_model, optimizer, global_step, epoch,
                )

    # ── Final save ──
    if accelerator.is_main_process:
        _save_checkpoint(
            args, accelerator, unet, measurement_encoder,
            image_proj_model, optimizer, global_step, epoch, final=True,
        )

    print("[8/8] Training complete!")
    accelerator.end_training()


# ─── Checkpoint Saving ───────────────────────────────────────────────────────

def _save_checkpoint(
    args, accelerator, unet, measurement_encoder, image_proj_model,
    optimizer, global_step, epoch, final=False,
):
    """Save LoRA adapter + MeasurementEncoder + Resampler (NOT full pipeline)."""
    tag = "final" if final else f"checkpoint-{global_step}"
    save_path = os.path.join(args.output_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    print(f"  Saving checkpoint to {save_path}...")

    # Save LoRA adapter
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_path = os.path.join(save_path, "lora_adapter")
    unwrapped_unet.save_pretrained(lora_path)

    # Save MeasurementEncoder
    me_state = accelerator.unwrap_model(measurement_encoder).state_dict()
    torch.save(me_state, os.path.join(save_path, "measurement_encoder.pt"))

    # Save Resampler (IP-adapter projection)
    res_state = accelerator.unwrap_model(image_proj_model).state_dict()
    torch.save(res_state, os.path.join(save_path, "resampler.pt"))

    # Save optimizer state for full resume
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))

    # Save training state for resume
    with open(os.path.join(save_path, "training_state.json"), "w") as f:
        json.dump(
            {"global_step": global_step, "epoch": epoch + 1},
            f,
            indent=2,
        )

    # Save args for reproducibility
    with open(os.path.join(save_path, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"  ✓ Checkpoint saved ({tag})")


# ─── Validation ──────────────────────────────────────────────────────────────

def _run_validation(
    args,
    accelerator,
    unet,
    vae,
    noise_scheduler,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    image_encoder,
    unet_encoder,
    test_dataloader,
    measurement_encoder,
    image_proj_model,
    global_step,
):
    """Run validation and save sample images with measurement conditioning."""
    print(f"\n  [Validation @ step {global_step}]")

    # ── Wrapper that prepends measurement tokens to Resampler output ──
    class MeasurementAwareResampler(torch.nn.Module):
        """Wraps the Resampler to also prepend measurement tokens during inference."""
        def __init__(self, resampler, meas_encoder, measurements_batch):
            super().__init__()
            self.resampler = resampler
            self.meas_encoder = meas_encoder
            self.measurements_batch = measurements_batch

        def forward(self, image_embeds):
            # image_embeds may be doubled for CFG: [uncond; cond]
            ip_tokens = self.resampler(image_embeds)  # (2B, 16, 2048)

            bsz = self.measurements_batch.shape[0]
            meas_tokens = self.meas_encoder(self.measurements_batch)
            meas_tokens = meas_tokens.to(dtype=ip_tokens.dtype)

            # If CFG is active, image_embeds is [uncond_batch; cond_batch]
            if ip_tokens.shape[0] == 2 * bsz:
                zero_meas = torch.zeros_like(meas_tokens)
                meas_tokens_cfg = torch.cat([zero_meas, meas_tokens], dim=0)
            else:
                meas_tokens_cfg = meas_tokens

            # Combine: [measurement_tokens | ip_tokens]
            return torch.cat([meas_tokens_cfg, ip_tokens], dim=1)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            unwrapped_unet = accelerator.unwrap_model(unet)

            # Merge LoRA for inference
            unwrapped_unet.merge_adapter()

            newpipe = TryonPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unwrapped_unet,
                vae=vae,
                scheduler=noise_scheduler,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                image_encoder=image_encoder,
                unet_encoder=unet_encoder,
                torch_dtype=torch.float16,
                add_watermarker=False,
                safety_checker=None,
            ).to(accelerator.device)

            for sample in test_dataloader:
                img_emb_list = [sample["cloth"][i] for i in range(sample["cloth"].shape[0])]
                prompt = sample["caption"]
                num_prompts = sample["cloth"].shape[0]
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts

                image_embeds_raw = torch.cat(img_emb_list, dim=0)

                # ── Inject measurement tokens via Resampler wrapper ──
                measurements = sample["measurements"].to(
                    accelerator.device, dtype=torch.float32
                )
                original_proj = newpipe.unet.encoder_hid_proj
                newpipe.unet.encoder_hid_proj = MeasurementAwareResampler(
                    image_proj_model, measurement_encoder, measurements
                )

                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = newpipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt_cloth = sample["caption_cloth"]
                    if not isinstance(prompt_cloth, List):
                        prompt_cloth = [prompt_cloth] * num_prompts

                    (prompt_embeds_c, _, _, _) = newpipe.encode_prompt(
                        prompt_cloth,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                    generator = (
                        torch.Generator(newpipe.device).manual_seed(args.seed)
                        if args.seed is not None
                        else None
                    )
                    images = newpipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=sample["pose_img"],
                        text_embeds_cloth=prompt_embeds_c,
                        cloth=sample["cloth_pure"].to(accelerator.device),
                        mask_image=sample["inpaint_mask"],
                        image=(sample["image"] + 1.0) / 2.0,
                        height=args.height,
                        width=args.width,
                        guidance_scale=args.guidance_scale,
                        ip_adapter_image=image_embeds_raw,
                    )[0]

                # Restore original encoder_hid_proj
                newpipe.unet.encoder_hid_proj = original_proj

                # ── Compute SSIM against ground truth ──
                ssim_scores = []
                for i, img in enumerate(images):
                    img.save(
                        os.path.join(
                            args.output_dir,
                            f"{global_step}_{i}_val.jpg",
                        )
                    )
                    # Compute SSIM between generated and ground truth
                    try:
                        from skimage.metrics import structural_similarity as ssim
                        gen_np = np.array(img.resize((args.width, args.height)))
                        gt_tensor = sample["image"][i]  # (-1, 1) normalized
                        gt_np = ((gt_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                        score = ssim(gt_np, gen_np, channel_axis=2, data_range=255)
                        ssim_scores.append(score)
                    except Exception:
                        pass

                avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
                print(f"  Validation SSIM: {avg_ssim:.4f} (n={len(ssim_scores)})")

                # Log metrics
                metrics_path = os.path.join(args.output_dir, "training_metrics.json")
                metrics = []
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                metrics.append({
                    "step": global_step,
                    "ssim": float(avg_ssim),
                    "num_samples": len(ssim_scores),
                })
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                break  # Only one batch for validation

            # Unmerge LoRA to resume training
            unwrapped_unet.unmerge_adapter()

            del newpipe
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
