#!/usr/bin/env bash
# 强化版预训练 ViT：训练增强 + 梯度裁剪 + 略减弱域/反事实权重 + 更偏向正类的 focal。
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-boost_vit_$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-logs/${TAG}}"
mkdir -p "${OUTDIR}/cp"

EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-5}"
VIT_BACKBONE_MULT="${VIT_BACKBONE_MULT:-0.1}"

python training/train_oct_traige.py \
  --encoder_type vit \
  --vit_pretrained 1 \
  --checkpoint_dir "${OUTDIR}/cp" \
  --log_dir "${OUTDIR}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --warmup_epochs "${WARMUP}" \
  --vit_backbone_lr_mult "${VIT_BACKBONE_MULT}" \
  --lambda_adv 0.25 \
  --lambda_ortho 0.35 \
  --lambda_consist 0.1 \
  --focal_alpha 0.5 \
  --focal_gamma 2.0 \
  --use_train_augment 1 \
  --max_grad_norm 1.0

echo "[done] checkpoints: ${OUTDIR}/cp/best_model.pt"
echo "[done] metrics csv under: ${OUTDIR}/metrics_history_*.csv"
