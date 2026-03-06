#!/bin/bash
export HF_HOME=/workspace/huggingface_cache
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
cd /workspace/FitFusion

echo "Starting IDM-VTON Size-Aware QLoRA Training..."
python -u IDM-VTON/train_xl_qlora.py \
  --pretrained_model_name_or_path /workspace/model_cache/IDM-VTON \
  --pretrained_garmentnet_path /workspace/model_cache/IDM-VTON \
  --pretrained_ip_adapter_path /workspace/model_cache/h94_IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin \
  --image_encoder_path /workspace/model_cache/IDM-VTON/image_encoder \
  --data_dir /workspace/FitFusion/data/fitfusion_vitonhd \
  --output_dir /workspace/FitFusion/output_qlora \
  --train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 50 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --lora_rank 16 \
  --lora_alpha 32 \
  --num_measurement_tokens 4 \
  --logging_steps 50 \
  --checkpointing_epoch 5


