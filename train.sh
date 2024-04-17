export MODEL_NAME="pretrained_models/stable-diffusion-v1-5"
export DATASET_PATH="CUHK-SYSU/cropped_image"

accelerate launch --config_file=accelerate_config.yaml --mixed_precision="fp16" \
  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_PATH \
  --h=256 --w=128 \
  --random_flip \
  --train_batch_size=44 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="PIS/model" \
  --checkpointing_steps=7000 \
  --checkpoints_total_limit=1 \
  --class_label