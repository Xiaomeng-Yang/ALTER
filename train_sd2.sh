export DATASET_NAME=your_hf_or_custom_dataset
export OUTPUT_DIR=outputs/alter_sd21
export VALIDATION_PROMPT="a high quality photo of a cat"
export PROJECT_NAME=ALTER_SD21

accelerate launch sd_prune.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --image_column="jpg" \
  --caption_column="json" \
  --train_batch_size=32 \
  --max_train_steps=32000 \
  --control_epochs=2 \
  --pruning_ratio=0.45 \
  --unet_learning_rate=1e-05 \
  --hypernet_lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=250 \
  --hypernet_learning_rate=5e-05 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt=$VALIDATION_PROMPT \
  --validation_epochs 1 \
  --checkpointing_steps=4000 \
  --output_dir=$OUTPUT_DIR \
  --distill \
  --distill_only \
  --feature_kd \
  --ratio_loss_rate=5.0 \
  --router_balance \
  --tracker_project_name=$PROJECT_NAME

