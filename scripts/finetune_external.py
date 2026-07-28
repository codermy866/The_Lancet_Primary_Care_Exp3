#!/usr/bin/env python3
"""在外部校准集上微调已有模型（小样本 few-epoch fine-tune）。

用法示例：
python3 finetune_external.py \
  --pretrained_checkpoint checkpoints/best_model.pt \
  --train_csv logs/external_calibration_set.csv \
  --val_csv logs/external_holdout_set.csv \
  --out_checkpoint_dir checkpoints/finetune_ext \
  --out_log_dir logs/finetune_ext \
  --epochs 3 --lr 1e-5 --use_train_augment 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.read_csv(p, encoding="gbk")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on external calibration set")
    parser.add_argument("--pretrained_checkpoint", type=str, required=False, default="checkpoints/best_model.pt")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--out_checkpoint_dir", type=str, default="checkpoints/finetune_ext")
    parser.add_argument("--out_log_dir", type=str, default="logs/finetune_ext")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_train_augment", type=int, default=1)
    parser.add_argument("--freeze_backbone", type=int, default=1, help="1 freeze oct_encoder backbone, 0 train all")
    parser.add_argument("--head_lr", type=float, default=None, help="If set, use this lr for trainable params (head)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 项目根（experiments/OCT_traige）加入 path，按现有代码风格
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    from config import OCTTraigeConfig
    from data.dataset_oct_only import OCTOnlyDataset, _extract_center_id_from_oct_id
    from models.oct_traige_model import OCTTraigeModel
    from training.train_oct_traige import train_one_epoch, validate_one_epoch, FocalLoss

    config = OCTTraigeConfig()
    config.num_epochs = int(args.epochs)
    config.learning_rate = float(args.lr)
    config.batch_size = int(args.batch_size)
    config.num_workers = int(args.num_workers)
    config.use_train_augment = bool(int(args.use_train_augment))
    config.checkpoint_dir = str(args.out_checkpoint_dir)
    config.log_dir = str(args.out_log_dir)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device} lr={config.learning_rate} epochs={config.num_epochs}")

    # 为了兼容 OCTOnlyDataset 的目录检查，创建一个临时 data_root（会被 dataset 忽略，因为 CSV 含 oct_paths）
    tmp_data_root = Path(config.log_dir) / "finetune_data_root"
    (tmp_data_root / "internal_train" / "train" / "oct").mkdir(parents=True, exist_ok=True)
    (tmp_data_root / "internal_train" / "val" / "oct").mkdir(parents=True, exist_ok=True)

    train_csv_p = Path(args.train_csv)
    val_csv_p = Path(args.val_csv)
    if not train_csv_p.exists() or not val_csv_p.exists():
        raise FileNotFoundError(f"train/val csv not found: {train_csv_p}, {val_csv_p}")

    # 构建 center_to_idx 映射，使 train/val 一致
    df_train = _safe_read_csv(train_csv_p)
    df_val = _safe_read_csv(val_csv_p)

    def _get_centers(df: pd.DataFrame) -> list:
        if "center_id" in df.columns:
            return sorted(df["center_id"].astype(str).unique().tolist())
        if "oct_id" in df.columns:
            return sorted({str(_extract_center_id_from_oct_id(x)) for x in df["oct_id"].values})
        return []

    centers = sorted(set(_get_centers(df_train) + _get_centers(df_val)))
    if not centers:
        raise ValueError("无法推断任何中心 ID")
    center_to_idx = {c: i for i, c in enumerate(centers)}
    print(f"[finetune] centers={len(center_to_idx)} mapping sample={list(center_to_idx.items())[:3]}")

    # 如果存在预训练 checkpoint，尝试读取其中的 memory_bank 或 center_discriminator 形状
    pretrained = Path(args.pretrained_checkpoint)
    pretrained_n_centers = None
    if pretrained.exists():
        try:
            ck_inspect = torch.load(pretrained, map_location="cpu")
            st = ck_inspect.get("model_state_dict", ck_inspect)
            if isinstance(st, dict):
                for k, v in st.items():
                    try:
                        if isinstance(v, torch.Tensor):
                            if k.endswith("memory_bank.bank"):
                                pretrained_n_centers = int(v.shape[0])
                                break
                            if k.endswith("center_discriminator.net.6.weight"):
                                pretrained_n_centers = int(v.shape[0])
                                break
                    except Exception:
                        continue
        except Exception as e:
            print(f"[finetune] warning: cannot inspect pretrained checkpoint: {e}")

    model_num_centers = max(len(center_to_idx), pretrained_n_centers or 0)
    if model_num_centers == 0:
        model_num_centers = len(center_to_idx)
    print(f"[finetune] model will use num_centers={model_num_centers} (train csv centers={len(center_to_idx)} pretrained_centers={pretrained_n_centers})")

    # dataset（使用 CSV 中的 oct_paths）
    train_ds = OCTOnlyDataset(
        csv_path=str(train_csv_p),
        data_root=str(tmp_data_root),
        split="train",
        oct_frames=config.oct_frames,
        img_size=config.img_size,
        center_to_idx=center_to_idx,
        train_augment=config.use_train_augment,
    )
    val_ds = OCTOnlyDataset(
        csv_path=str(val_csv_p),
        data_root=str(tmp_data_root),
        split="val",
        oct_frames=config.oct_frames,
        img_size=config.img_size,
        center_to_idx=center_to_idx,
        train_augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = OCTTraigeModel(
        embed_dim=config.embed_dim,
        num_classes=2,
        oct_num_slices=config.oct_frames,
        dropout=config.dropout,
        num_centers=model_num_centers,
        memory_capacity=config.memory_capacity,
        alpha_cf=config.alpha_cf,
        encoder_type=config.encoder_type,
        vit_pretrained=config.vit_pretrained,
        img_size=config.img_size,
    ).to(device)

    # load pretrained
    pretrained = Path(args.pretrained_checkpoint)
    if pretrained.exists():
        ck = torch.load(pretrained, map_location=device)
        state = ck.get("model_state_dict", ck)

        # 兼容性加载：只加载与当前模型形状一致的参数
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        if isinstance(state, dict):
            for k, v in state.items():
                if k in model_state:
                    try:
                        if isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
                            model_state[k] = v
                            loaded_keys.append(k)
                        else:
                            skipped_keys.append(k)
                    except Exception:
                        skipped_keys.append(k)
                else:
                    skipped_keys.append(k)

        # 将更新后的 state_dict 写入模型（此处 keys 与模型完全匹配）
        model.load_state_dict(model_state)
        print(f"[finetune] loaded pretrained from {pretrained} (loaded_keys={len(loaded_keys)}, skipped_keys={len(skipped_keys)})")
        if skipped_keys:
            print(f"[finetune] sample skipped keys: {skipped_keys[:8]}")
    else:
        print(f"[finetune] pretrained checkpoint not found, training from scratch: {pretrained}")

    # 可选：冻结 backbone，仅训练 head/少数参数，防止小样本微调时灾难性遗忘
    if int(args.freeze_backbone):
        try:
            for p in model.oct_encoder.parameters():
                p.requires_grad = False
        except Exception:
            pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        # fallback: 若没有 trainable params，退回到全部参数
        trainable_params = model.parameters()

    opt_lr = float(config.learning_rate) if args.head_lr is None else float(args.head_lr)
    optimizer = torch.optim.AdamW(trainable_params, lr=opt_lr, weight_decay=config.weight_decay)
    # criterion
    if config.use_focal_loss:
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_path = Path(config.checkpoint_dir) / "best_model_finetune.pt"
    last_path = Path(config.checkpoint_dir) / "last_model_finetune.pt"
    history = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device, config=config)
        val_metrics = validate_one_epoch(model=model, loader=val_loader, criterion=criterion, device=device, config=config)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print(f"Epoch {epoch}/{config.num_epochs} val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, best_path)
            print(f"  [finetune] Saved best model to {best_path} (AUC={best_auc:.4f})")

    torch.save({"model_state_dict": model.state_dict(), "epoch": config.num_epochs, "best_auc": best_auc}, last_path)
    hist_json = Path(config.log_dir) / f"finetune_history.json"
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[finetune] Done. Best AUC={best_auc:.4f}. Saved best={best_path} last={last_path} history={hist_json}")


if __name__ == "__main__":
    main()
