from safetensors.torch import load_file
import sys

print('Loading safetensors...')
try:
    state_dict = load_file('/workspace/model_cache/IDM-VTON/unet_encoder/diffusion_pytorch_model.safetensors')
    print('Loaded keys:', len(state_dict))
except Exception as e:
    with open('error.txt', 'w') as f:
        f.write(str(e))
    sys.exit(1)
