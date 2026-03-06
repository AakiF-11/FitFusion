import torch
from diffusers import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

print('Loading unet_encoder...')
try:
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        '/workspace/model_cache/IDM-VTON',
        subfolder='unet_encoder',
        use_safetensors=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print('Loaded successfully!')
except Exception as e:
    import traceback
    traceback.print_exc()

print('Loading unet...')
try:
    unet = UNet2DConditionModel.from_pretrained(
        '/workspace/model_cache/IDM-VTON',
        subfolder='unet',
        low_cpu_mem_usage=True,
        device_map=None,
        use_safetensors=True,
        torch_dtype=torch.float16
    )
    print('Loaded successfully!')
except Exception as e:
    import traceback
    traceback.print_exc()
