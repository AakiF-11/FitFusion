"""
src/inference/tryon.py
Executes a single size-aware try-on pass through the IDM-VTON pipeline.
Accepts pre-processed inputs; returns a PIL Image.
"""
import torch
from PIL import Image
from typing import Any
from torchvision import transforms

_TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


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
    Follows the call pattern from IDM-VTON/gradio_demo/app.py exactly.
    """
    device = pipeline.device

    person_image = Image.open(person_image_path).convert("RGB").resize((width, height))

    # Resize inputs to target resolution
    garment_pil = warped_garment.convert("RGB").resize((width, height))
    pose_pil = densepose_map.convert("RGB").resize((width, height))

    # Encode text prompts
    pos_prompt = physics_params.positive_prompt
    neg_prompt = getattr(
        physics_params, "negative_prompt",
        "monochrome, lowres, bad anatomy, worst quality, low quality"
    )

    with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            pos_prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg_prompt,
        )
        # Cloth text embeddings: describes the garment image
        (prompt_embeds_c, _, _, _) = pipeline.encode_prompt(
            "a photo of a garment",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    # Convert images to normalized float16 tensors
    pose_tensor = _TENSOR_TRANSFORM(pose_pil).unsqueeze(0).to(device, torch.float16)
    cloth_tensor = _TENSOR_TRANSFORM(garment_pil).unsqueeze(0).to(device, torch.float16)

    generator = torch.Generator(device=str(device)).manual_seed(seed)

    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            images = pipeline(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=cloth_tensor,
                mask_image=agnostic_mask,
                image=person_image,
                height=height,
                width=width,
                ip_adapter_image=garment_pil,
                guidance_scale=guidance_scale,
            )[0]

    return images[0]
