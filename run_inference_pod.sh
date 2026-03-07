#!/bin/bash
cd /workspace/FitFusion
git pull origin master
cd IDM-VTON
python run_tryon.py \
  --product_id "cropped-borg-aviator-jacket-black" \
  --brand_id "snag_tights" \
  --s3_image_url "file:///workspace/FitFusion/data/test_inputs/person_model_C_Pixie.jpg" \
  --bust 92 \
  --waist 76 \
  --hips 100 \
  --height 167 \
  --all_sizes \
  --output /workspace/output_all_sizes.png \
  --steps 30 \
  --seed 42 \
  2>&1 | tee /workspace/inference_log.txt
