export RESULTS_DIR=outputs/alter_dit_xl2

torchrun --nnodes=1 --nproc_per_node=2 dit_prune.py \
  --model DiT-XL/2 \
  --load-weight DiT-XL-2-256x256.pt \
  --data-path dataset/imagenet_encoded \
  --results-dir $RESULTS_DIR \
  --epochs 80 \
  --control_epochs 20 \
  --router_balance \
  --ratio_loss_rate=5.0 \
  --hyper_lr=5e-4 \
  --global-batch-size 256 \
  --distill \
  --distill_only \
  --feature_kd
