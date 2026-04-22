#!/usr/bin/env bash
# CNN vs 预训练 ViT 对照：独立 checkpoint/log，最后生成汇总 CSV。
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-logs/encoder_compare_${TAG}}"
mkdir -p "${OUTDIR}/cnn" "${OUTDIR}/vit_pt"

EPOCHS="${EPOCHS:-30}"
# CNN：沿用原默认量级
LR_CNN="${LR_CNN:-2e-4}"
WARMUP_CNN="${WARMUP_CNN:-3}"
# ViT+ImageNet：更小 base lr + 更长 warmup + backbone 分组（见 train_oct_traige 默认）
LR_VIT="${LR_VIT:-5e-5}"
WARMUP_VIT="${WARMUP_VIT:-5}"
VIT_BACKBONE_MULT="${VIT_BACKBONE_MULT:-0.1}"

echo ">>> [1/2] CNN encoder  (${OUTDIR}/cnn)"
python training/train_oct_traige.py \
  --encoder_type cnn \
  --checkpoint_dir "${OUTDIR}/cnn" \
  --log_dir "${OUTDIR}/cnn" \
  --lr "${LR_CNN}" \
  --warmup_epochs "${WARMUP_CNN}" \
  --epochs "${EPOCHS}"

echo ">>> [2/2] ViT (ImageNet pretrained) (${OUTDIR}/vit_pt)"
python training/train_oct_traige.py \
  --encoder_type vit \
  --vit_pretrained 1 \
  --vit_backbone_lr_mult "${VIT_BACKBONE_MULT}" \
  --checkpoint_dir "${OUTDIR}/vit_pt" \
  --log_dir "${OUTDIR}/vit_pt" \
  --lr "${LR_VIT}" \
  --warmup_epochs "${WARMUP_VIT}" \
  --epochs "${EPOCHS}"

echo ">>> Summary -> ${OUTDIR}/encoder_compare_summary.csv"
python scripts/summarize_encoder_compare.py "${OUTDIR}" --out_csv "${OUTDIR}/encoder_compare_summary.csv"
