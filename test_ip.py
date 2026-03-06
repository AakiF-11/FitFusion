import os
import torch
import traceback

path = '/workspace/model_cache/IDM-VTON/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin'
print(f"Size of IP-Adapter: {os.path.getsize(path)}")

print('Loading IP-Adapter...')
try:
    state_dict = torch.load(path, map_location='cpu')
    print('Keys:', len(state_dict))
except Exception as e:
    with open('error_ip.txt', 'w') as f:
        f.write(str(e))
    traceback.print_exc()
