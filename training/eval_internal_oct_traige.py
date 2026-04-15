"""
在 best checkpoint 上对内部 train / val 集推理，并按医院 (center_id) 汇总二分类指标。
与训练日志中的 epoch 指标可能略有差异（此处为 eval 模式、无 dropout）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import OCTTraigeConfig
from data.dataset_oct_only import OCTOnlyDataset
from models.oct_traige_model import OCTTraigeModel
from training.train_oct_traige import _build_center_mapping, _compute_binary_metrics


@torch.no_grad()
def _run_split(
    *,
    csv_path: Path,
    data_root: Path,
    split: str,
    center_to_idx: dict,
    model: nn.Module,
    device: torch.device,
    config: OCTTraigeConfig,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ds = OCTOnlyDataset(
        csv_path=str(csv_path),
        data_root=str(data_root),
        split=split,
        oct_frames=config.oct_frames,
        img_size=config.img_size,
        center_to_idx=center_to_idx,
    )
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        oct_images = batch["oct_images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        center_idx = batch["center_idx"].to(device, non_blocking=True)
        outputs = model(oct_images=oct_images, center_labels=center_idx, return_loss_components=False)
        logits = outputs["pred"]
        loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu().item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    overall = _compute_binary_metrics(logits_cat, labels_cat)
    overall["loss"] = total_loss / max(len(loader), 1)

    df = pd.read_csv(csv_path, encoding="utf-8").reset_index(drop=True)
    probs = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    preds = logits_cat.argmax(dim=1).numpy()
    df["pred"] = preds
    df["prob_pos"] = probs

    rows = []
    for center, g in df.groupby("center_id"):
        idx = g.index.to_list()
        m = _compute_binary_metrics(logits_cat[idx], labels_cat[idx])
        m["center_id"] = str(center)
        m["n"] = int(len(g))
        rows.append(m)
    per_center = pd.DataFrame(rows).sort_values("center_id")

    return overall, per_center, df


def main():
    parser = argparse.ArgumentParser(description="Internal train/val per-centre metrics (best checkpoint)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out_train_csv", type=str, default="logs/internal_train_per_center_metrics_loc5out.csv")
    parser.add_argument("--out_val_csv", type=str, default="logs/internal_val_per_center_metrics_loc5out.csv")
    parser.add_argument("--out_json", type=str, default="logs/internal_overall_metrics_loc5out.json")
    args = parser.parse_args()

    config = OCTTraigeConfig()
    config.data_root = args.data_root
    data_root = Path(config.data_root)
    train_csv = data_root / config.train_csv_name
    val_csv = data_root / config.val_csv_name
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("缺少 train_labels.csv 或 val_labels.csv")

    center_to_idx = _build_center_mapping(train_csv, val_csv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["model_state_dict"]
    disc_w = sd.get("center_discriminator.net.6.weight")
    if disc_w is None:
        raise KeyError("checkpoint 缺少 center_discriminator.net.6.weight")
    num_centers_ckpt = int(disc_w.shape[0])

    model = OCTTraigeModel(
        embed_dim=config.embed_dim,
        num_classes=2,
        oct_num_slices=config.oct_frames,
        dropout=config.dropout,
        num_centers=num_centers_ckpt,
        memory_capacity=config.memory_capacity,
        alpha_cf=config.alpha_cf,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    out_root = ROOT
    train_overall, train_pc, train_pred = _run_split(
        csv_path=train_csv,
        data_root=data_root,
        split="train",
        center_to_idx=center_to_idx,
        model=model,
        device=device,
        config=config,
    )
    val_overall, val_pc, val_pred = _run_split(
        csv_path=val_csv,
        data_root=data_root,
        split="val",
        center_to_idx=center_to_idx,
        model=model,
        device=device,
        config=config,
    )

    out_train = out_root / args.out_train_csv
    out_val = out_root / args.out_val_csv
    out_json = out_root / args.out_json
    out_train.parent.mkdir(parents=True, exist_ok=True)

    train_pc.to_csv(out_train, index=False, encoding="utf-8")
    val_pc.to_csv(out_val, index=False, encoding="utf-8")
    train_pred.to_csv(out_train.parent / "internal_train_predictions.csv", index=False, encoding="utf-8")
    val_pred.to_csv(out_train.parent / "internal_val_predictions.csv", index=False, encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_overall": train_overall,
                "val_overall": val_overall,
                "n_train": int(len(train_pred)),
                "n_val": int(len(val_pred)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[internal train] n={len(train_pred)} overall={train_overall}")
    print(f"[saved] {out_train}")
    print(f"[internal val]   n={len(val_pred)} overall={val_overall}")
    print(f"[saved] {out_val}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
