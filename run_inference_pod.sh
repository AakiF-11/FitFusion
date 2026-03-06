#!/bin/bash
cd /workspace/FitFusion
git pull origin master
python run_pipeline.py \
  --person_image data/test_inputs/person_model_C_Pixie.jpg \
  --garment_image data/test_inputs/garment_cropped_borg_aviator_jacket_black.png \
  --person_size M \
  --target_size XL \
  --garment_type top \
  --ckpt_dir ./ckpt \
  --output_dir ./output \
  --steps 30 \
  --guidance_scale 2.0 \
  --seed 42 \
  --device cuda \
  --skip_preprocessing \
  2>&1 | tee /workspace/inference_log.txt
