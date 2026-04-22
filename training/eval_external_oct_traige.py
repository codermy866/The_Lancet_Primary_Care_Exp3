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
from data.dataset_oct_only import OCTOnlyDataset, _extract_center_id_from_oct_id
from models.oct_traige_model import OCTTraigeModel
from training.train_oct_traige import _build_center_mapping, _compute_binary_metrics


def _build_external_oct_paths(ext_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """
    Ensure external dataframe has `oct_paths` so dataset loader can read real images.
    """
    if "oct_paths" in ext_df.columns and ext_df["oct_paths"].astype(str).str.len().gt(0).any():
        return ext_df

    if "oct_id" not in ext_df.columns:
        raise ValueError("external csv 缺少 oct_id/OCT，无法构建 oct_paths")

    oct_root_candidates = [
        data_root / "external_validation" / "oct",
        data_root / "external_test" / "oct",
        data_root / "external" / "oct",
    ]
    oct_root = None
    for cand in oct_root_candidates:
        if cand.exists():
            oct_root = cand
            break
    if oct_root is None:
        raise FileNotFoundError(
            f"未找到 external OCT 根目录，候选为: {[str(x) for x in oct_root_candidates]}"
        )

    oct_paths_col: list[str] = []
    missing = 0
    for oid in ext_df["oct_id"].astype(str).tolist():
        case_dir = oct_root / oid
        if not case_dir.exists():
            oct_paths_col.append("")
            missing += 1
            continue
        paths = sorted([str(p) for p in case_dir.glob("*.png")] + [str(p) for p in case_dir.glob("*.jpg")])
        oct_paths_col.append(";".join(paths))

    ext_df = ext_df.copy()
    ext_df["oct_paths"] = oct_paths_col
    if missing > 0:
        print(f"[warn] external cases missing OCT directory: {missing}/{len(ext_df)}")
    return ext_df


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate OCT_traige on external set")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--external_csv", type=str, default="external_test_labels.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out_json", type=str, default="logs/external_metrics.json")
    parser.add_argument("--out_csv", type=str, default="logs/external_per_center_metrics.csv")
    args = parser.parse_args()

    config = OCTTraigeConfig()
    config.data_root = args.data_root
    data_root = Path(config.data_root)

    train_csv = data_root / config.train_csv_name
    val_csv = data_root / config.val_csv_name
    ext_csv = data_root / args.external_csv
    if not train_csv.exists() or not val_csv.exists() or not ext_csv.exists():
        raise FileNotFoundError("train/val/external csv 文件不存在，请先完成数据准备")

    center_to_idx = _build_center_mapping(train_csv, val_csv)
    ext_df = pd.read_csv(ext_csv, encoding="utf-8")
    if "center_id" not in ext_df.columns:
        if "oct_id" not in ext_df.columns and "OCT" in ext_df.columns:
            ext_df = ext_df.rename(columns={"OCT": "oct_id"})
        if "oct_id" not in ext_df.columns:
            raise ValueError(f"{ext_csv} 缺少 center_id，且无法从 oct_id/OCT 推断中心")
        ext_df["center_id"] = ext_df["oct_id"].apply(_extract_center_id_from_oct_id)
    ext_df = _build_external_oct_paths(ext_df, data_root)
    ext_df["center_id_external"] = ext_df["center_id"].astype(str)
    # 推理时 forward(..., return_loss_components=False) 不使用 center_labels；
    # 但 checkpoint 的 discriminator/memory_bank 维度与训练时中心数一致，不能为外部中心扩容。
    dummy_center = sorted(center_to_idx.keys())[0]
    ext_df["center_id"] = dummy_center

    eval_csv = data_root / "_external_eval_tmp.csv"
    ext_df.to_csv(eval_csv, index=False, encoding="utf-8")

    ds = OCTOnlyDataset(
        csv_path=str(eval_csv),
        data_root=str(data_root),
        split="val",
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["model_state_dict"]
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    disc_w = sd.get("center_discriminator.net.6.weight")
    if disc_w is None:
        raise KeyError("checkpoint 缺少 center_discriminator.net.6.weight，无法推断 num_centers")
    num_centers_ckpt = int(disc_w.shape[0])

    model = OCTTraigeModel(
        embed_dim=config.embed_dim,
        num_classes=2,
        oct_num_slices=config.oct_frames,
        dropout=config.dropout,
        num_centers=num_centers_ckpt,
        memory_capacity=config.memory_capacity,
        alpha_cf=config.alpha_cf,
        encoder_type=str(ckpt_config.get("encoder_type", getattr(config, "encoder_type", "cnn"))),
        vit_pretrained=bool(ckpt_config.get("vit_pretrained", False)),
        img_size=int(ckpt_config.get("img_size", config.img_size)),
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_logits = []
    all_labels = []
    all_centers = []
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
        all_centers.extend(batch["oct_id"])

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    overall = _compute_binary_metrics(logits_cat, labels_cat)
    overall["loss"] = total_loss / max(len(loader), 1)

    pred_df = ext_df.copy().reset_index(drop=True)
    probs = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    preds = logits_cat.argmax(dim=1).numpy()
    pred_df["pred"] = preds
    pred_df["prob_pos"] = probs

    per_center_rows = []
    site_col = "center_id_external" if "center_id_external" in pred_df.columns else "center_id"
    for center, g in pred_df.groupby(site_col):
        idx = g.index.to_list()
        c_metrics = _compute_binary_metrics(logits_cat[idx], labels_cat[idx])
        c_metrics["center_id"] = str(center)
        c_metrics["n"] = int(len(g))
        per_center_rows.append(c_metrics)
    per_center_df = pd.DataFrame(per_center_rows).sort_values("center_id")

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_pred = out_json.parent / "external_predictions.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "num_samples": int(len(pred_df))}, f, ensure_ascii=False, indent=2)
    per_center_df.to_csv(out_csv, index=False, encoding="utf-8")
    pred_df.to_csv(out_pred, index=False, encoding="utf-8")

    print(f"[external] n={len(pred_df)} metrics={overall}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_pred}")


if __name__ == "__main__":
    main()

