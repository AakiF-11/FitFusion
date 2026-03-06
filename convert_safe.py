import torch
from safetensors.torch import save_file
import gc

bin_path = '/workspace/model_cache/IDM-VTON/unet/diffusion_pytorch_model.bin'
safe_path = '/workspace/model_cache/IDM-VTON/unet/diffusion_pytorch_model.safetensors'

print('Loading bin with mmap=True...')
try:
    state_dict = torch.load(bin_path, map_location='cpu', mmap=True)
    print('Dict loaded! Converting to Safetensors...')
    save_file(state_dict, safe_path)
    print('Successfully saved safetensors! Size:', len(state_dict))
except Exception as e:
    import traceback
    traceback.print_exc()
