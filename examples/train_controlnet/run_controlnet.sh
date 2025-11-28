export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="/home/dszh/workspace/tmp-smoke/data/controlnet"
export OUTPUT_DIR="/home/dszh/workspace/tmp-smoke/models/controlnet-finetuned"

accelerate launch --mixed_precision="fp16"  train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name_or_path=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=8 \
  --num_train_epochs=3 \
  --learning_rate=1e-5 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --validation_ids 1500 5500 8500 \
  --checkpoints_total_limit=0 \
  --dataloader_num_workers=8 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --trust_remote_code