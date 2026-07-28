#!/usr/bin/env python3
"""在外部 CSV 上用指定 checkpoint 生成概率（prob_pos）并保存为新的 external_predictions_*.csv。

用法示例：
python3 predict_external.py \
  --checkpoint checkpoints/finetune_aug/best_model_finetune.pt \
  --csv logs/external_predictions.csv \
  --out logs/external_predictions_finetune_aug.csv \
  --batch_size 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import OCTTraigeConfig
from data.dataset_oct_only import OCTOnlyDataset, _extract_center_id_from_oct_id
from models.oct_traige_model import OCTTraigeModel


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.read_csv(p, encoding="gbk")


def inspect_checkpoint_n_centers(p: Path):
    try:
        ck = torch.load(p, map_location="cpu")
    except Exception:
        return None
    st = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if not isinstance(st, dict):
        return None
    for k, v in st.items():
        try:
            if isinstance(v, torch.Tensor):
                if k.endswith("memory_bank.bank"):
                    return int(v.shape[0])
                if k.endswith("center_discriminator.net.6.weight"):
                    return int(v.shape[0])
        except Exception:
            continue
    return None


def load_checkpoint_shape_safe(model: torch.nn.Module, ck_path: Path, device: torch.device):
    ck = torch.load(ck_path, map_location=device)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    try:
        model.load_state_dict(state, strict=False)
        return state, [], []
    except Exception:
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
        model.load_state_dict(model_state)
        return state, loaded_keys, skipped_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    ck_p = Path(args.checkpoint)
    csv_p = Path(args.csv)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_p}")
    df = _safe_read_csv(csv_p)

    # estimate centers from CSV
    def _get_centers_from_df(df):
        if "center_id" in df.columns:
            return sorted(df["center_id"].astype(str).unique().tolist())
        if "oct_id" in df.columns:
            return sorted({str(_extract_center_id_from_oct_id(x)) for x in df["oct_id"].values})
        return []

    df_centers = _get_centers_from_df(df)
    n_centers_csv = len(df_centers)

    pretrained_n_centers = inspect_checkpoint_n_centers(ck_p) if ck_p.exists() else None
    model_num_centers = max(n_centers_csv, pretrained_n_centers or 0)
    if model_num_centers == 0:
        model_num_centers = max(1, n_centers_csv)

    print(f"[predict] csv centers={n_centers_csv} pretrained_centers={pretrained_n_centers} -> model_num_centers={model_num_centers}")

    # build tmp data root to satisfy OCTOnlyDataset directory checks
    tmp_data_root = out_p.parent / "predict_data_root"
    (tmp_data_root / "internal_train" / "train" / "oct").mkdir(parents=True, exist_ok=True)
    (tmp_data_root / "internal_train" / "val" / "oct").mkdir(parents=True, exist_ok=True)

    # construct center_to_idx mapping from CSV so dataset center_idx are valid
    center_to_idx = {c: i for i, c in enumerate(df_centers)} if df_centers else None

    # dataset
    cfg = OCTTraigeConfig()
    ds = OCTOnlyDataset(csv_path=str(csv_p), data_root=str(tmp_data_root), split="val", oct_frames=cfg.oct_frames, img_size=cfg.img_size, center_to_idx=center_to_idx, train_augment=False)

    loader = torch.utils.data.DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # build model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = OCTTraigeModel(
        embed_dim=cfg.embed_dim,
        num_classes=2,
        oct_num_slices=cfg.oct_frames,
        dropout=cfg.dropout,
        num_centers=int(model_num_centers),
        memory_capacity=cfg.memory_capacity,
        alpha_cf=cfg.alpha_cf,
        encoder_type=cfg.encoder_type,
        vit_pretrained=cfg.vit_pretrained,
        img_size=cfg.img_size,
    ).to(device)

    if ck_p.exists():
        state, loaded_keys, skipped_keys = load_checkpoint_shape_safe(model, ck_p, device)
        print(f"[predict] loaded checkpoint {ck_p} (loaded_keys={len(loaded_keys)}, skipped_keys={len(skipped_keys)})")
        if skipped_keys:
            print("[predict] sample skipped keys:", skipped_keys[:8])
    else:
        print(f"[predict] checkpoint not found: {ck_p}, running with random weights")

    # inference
    model.eval()
    probs = []
    preds = []
    labels = []
    oct_ids = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["oct_images"].to(device)
            lbl = batch["label"].cpu().numpy()
            center_idx = batch.get("center_idx")
            if center_idx is not None:
                center_idx = center_idx.to(device)
            outputs = model(oct_images=imgs, center_labels=center_idx, return_loss_components=False)
            logits = outputs["pred"]
            prob_pos = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            probs.extend(prob_pos.tolist())
            preds.extend((prob_pos >= 0.5).astype(int).tolist())
            labels.extend(lbl.tolist())
            # oct_id may be list of strings
            oids = batch.get("oct_id")
            if isinstance(oids, (list, tuple)):
                oct_ids.extend([str(x) for x in oids])
            else:
                oct_ids.extend([str(x) for x in oids.tolist()])

    # align length
    assert len(oct_ids) == len(df), f"length mismatch: loader produced {len(oct_ids)} rows but csv has {len(df)}"

    out_df = df.copy()
    # write standardized prob column name 'prob_pos'
    out_df["prob_pos"] = probs
    out_df["pred"] = preds

    out_df.to_csv(out_p, index=False)
    print(f"[predict] saved predictions to {out_p} (n={len(out_df)})")


if __name__ == '__main__':
    main()
