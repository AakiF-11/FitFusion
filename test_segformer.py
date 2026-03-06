import torch
import cv2
from PIL import Image
from transformers import pipeline

print('loading model...')
try:
    p = pipeline('image-segmentation', model='mattmdjaga/segformer_b2_clothes', device=0)
    print('inferring...')
    img = Image.new('RGB', (512, 512))
    res = p(img)
    print('success:', len(res))
except Exception as e:
    import traceback
    traceback.print_exc()
