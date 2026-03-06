from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('yisol/IDM-VTON')
print([f for f in files if 'unet/' in f and not 'unet_encoder' in f])
