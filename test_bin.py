import torch
import traceback
print('Testing torch.load...')
try:
    state_dict = torch.load('/workspace/model_cache/IDM-VTON/unet/diffusion_pytorch_model.bin', map_location='cpu')
    print('Keys:', len(state_dict))
except Exception as e:
    with open('error_bin.txt', 'w') as f:
        f.write(str(e))
