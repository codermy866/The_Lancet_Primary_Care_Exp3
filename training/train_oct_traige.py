from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]  # experiments/OCT_traige
sys.path.insert(0, str(ROOT))

from config import OCTTraigeConfig
from data.dataset_oct_only import OCTOnlyDataset, _extract_center_id_from_oct_id
from models.oct_traige_model import OCTTraigeModel


class FocalLoss(nn.Module):
    """
    二分类 focal loss（基于 softmax + 交叉熵的通用形式）
    focal_alpha：正类权重；负类为 1-alpha
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 2], targets: [B]
        log_probs = F.log_softmax(logits, dim=1)
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = log_pt.exp()  # [B]

        ce = -log_pt  # [B]
        alpha_t = torch.where(targets == 1, torch.tensor(self.alpha, device=logits.device), torch.tensor(1.0 - self.alpha, device=logits.device))
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _build_center_mapping(train_csv: Path, val_csv: Path) -> dict[str, int]:
    def _load_center_ids(csv_path: Path) -> list[str]:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(csv_path, encoding="gbk")

        if "center_id" in df.columns:
            return sorted(df["center_id"].astype(str).unique().tolist())

        if "oct_id" not in df.columns and "OCT" in df.columns:
            df = df.rename(columns={"OCT": "oct_id"})

        if "oct_id" not in df.columns:
            raise ValueError(f"CSV 缺少 `oct_id`/`OCT` 列：{csv_path}")

        centers = sorted({str(x) for x in df["oct_id"].apply(_extract_center_id_from_oct_id).unique().tolist()})
        return centers

    centers = sorted(set(_load_center_ids(train_csv) + _load_center_ids(val_csv)))
    if not centers:
        raise ValueError("未能从 train/val CSV 推断到任何 center_id")
    return {c: i for i, c in enumerate(centers)}


def _compute_binary_metrics(logits_cat: torch.Tensor, labels_cat: torch.Tensor) -> dict:
    probs_pos = F.softmax(logits_cat, dim=1)[:, 1].numpy()
    preds = logits_cat.argmax(dim=1).numpy()
    labels_np = labels_cat.numpy()

    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, zero_division=0)
    precision = precision_score(labels_np, preds, zero_division=0)
    recall = recall_score(labels_np, preds, zero_division=0)  # sensitivity
    bal_acc = balanced_accuracy_score(labels_np, preds)
    mcc = matthews_corrcoef(labels_np, preds) if len(np.unique(labels_np)) > 1 else 0.0

    try:
        auc = roc_auc_score(labels_np, probs_pos)
    except Exception:
        auc = 0.0
    try:
        pr_auc = average_precision_score(labels_np, probs_pos)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 同 precision

    return {
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "ppv": float(ppv),
        "npv": float(npv),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "balanced_acc": float(bal_acc),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
    }


def train_one_epoch(model: OCTTraigeModel, loader: DataLoader, optimizer, criterion, device, config: OCTTraigeConfig):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="train", leave=False):
        oct_images = batch["oct_images"].to(device, non_blocking=True)  # [B, S, 3, H, W]
        labels = batch["label"].to(device, non_blocking=True)
        center_labels = batch["center_idx"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            oct_images=oct_images,
            center_labels=center_labels,
            return_loss_components=True,
        )

        logits = outputs["pred"]
        L_cls = criterion(logits, labels)

        loss = config.lambda_cls * L_cls
        if "loss_components" in outputs:
            comps = outputs["loss_components"]
            if "L_adv" in comps:
                loss = loss + config.lambda_adv * comps["L_adv"]
            if "L_ortho" in comps:
                loss = loss + config.lambda_ortho * comps["L_ortho"]
            if "L_consist" in comps:
                loss = loss + config.lambda_consist * comps["L_consist"]

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = _compute_binary_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def validate_one_epoch(model: OCTTraigeModel, loader: DataLoader, criterion, device, config: OCTTraigeConfig):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="val", leave=False):
        oct_images = batch["oct_images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        center_labels = batch["center_idx"].to(device, non_blocking=True)

        outputs = model(
            oct_images=oct_images,
            center_labels=center_labels,
            return_loss_components=False,
        )
        logits = outputs["pred"]
        loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu().item())

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = _compute_binary_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="OCT_traige - OCT-only training")
    parser.add_argument("--data_root", type=str, default="", help="覆盖 config.data_root")
    parser.add_argument("--train_csv", type=str, default="", help="覆盖 train csv 文件名")
    parser.add_argument("--val_csv", type=str, default="", help="覆盖 val csv 文件名")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 num_epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="覆盖 checkpoint_dir")
    parser.add_argument("--log_dir", type=str, default="", help="覆盖 log_dir")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 learning_rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="覆盖 weight_decay")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖 dropout")
    parser.add_argument("--lambda_adv", type=float, default=None, help="覆盖 lambda_adv")
    parser.add_argument("--lambda_ortho", type=float, default=None, help="覆盖 lambda_ortho")
    parser.add_argument("--lambda_consist", type=float, default=None, help="覆盖 lambda_consist")
    parser.add_argument("--alpha_cf", type=float, default=None, help="覆盖 alpha_cf")
    parser.add_argument("--use_focal_loss", type=int, default=None, help="1启用 focal loss，0关闭")
    parser.add_argument("--focal_alpha", type=float, default=None, help="覆盖 focal_alpha")
    parser.add_argument("--focal_gamma", type=float, default=None, help="覆盖 focal_gamma")
    args = parser.parse_args()

    config = OCTTraigeConfig()
    if args.data_root:
        config.data_root = args.data_root
    if args.train_csv:
        config.train_csv_name = args.train_csv
    if args.val_csv:
        config.val_csv_name = args.val_csv
    if args.epochs is not None:
        config.num_epochs = int(args.epochs)
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.lr is not None:
        config.learning_rate = float(args.lr)
    if args.weight_decay is not None:
        config.weight_decay = float(args.weight_decay)
    if args.dropout is not None:
        config.dropout = float(args.dropout)
    if args.lambda_adv is not None:
        config.lambda_adv = float(args.lambda_adv)
    if args.lambda_ortho is not None:
        config.lambda_ortho = float(args.lambda_ortho)
    if args.lambda_consist is not None:
        config.lambda_consist = float(args.lambda_consist)
    if args.alpha_cf is not None:
        config.alpha_cf = float(args.alpha_cf)
    if args.use_focal_loss is not None:
        config.use_focal_loss = bool(int(args.use_focal_loss))
    if args.focal_alpha is not None:
        config.focal_alpha = float(args.focal_alpha)
    if args.focal_gamma is not None:
        config.focal_gamma = float(args.focal_gamma)

    for d in [config.checkpoint_dir, config.log_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OCT_traige] device={device} data_root={config.data_root}")

    data_root = Path(config.data_root)
    train_csv = data_root / config.train_csv_name
    val_csv = data_root / config.val_csv_name
    if not train_csv.exists():
        raise FileNotFoundError(f"缺少 train_csv: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"缺少 val_csv: {val_csv}")

    # 先从 CSV 推断中心映射（训练/验证必须一致）
    center_to_idx = _build_center_mapping(train_csv, val_csv)
    num_centers = len(center_to_idx)
    print(f"[OCT_traige] centers={num_centers} mapping={center_to_idx}")

    # Data transforms（不做增强，避免影响复现实验；需要增强可在这里加）
    # 注意：Dataset 内已实现 Resize/Normalize，因此这里不传 transform

    # 构建 train/val dataset
    train_ds = OCTOnlyDataset(
        csv_path=str(train_csv),
        data_root=str(config.data_root),
        split="train",
        oct_frames=config.oct_frames,
        img_size=config.img_size,
        center_to_idx=center_to_idx,
    )
    val_ds = OCTOnlyDataset(
        csv_path=str(val_csv),
        data_root=str(config.data_root),
        split="val",
        oct_frames=config.oct_frames,
        img_size=config.img_size,
        center_to_idx=center_to_idx,
    )

    # 训练集加权采样（按 label ）
    labels = train_ds.df["label"].astype(int).values if hasattr(train_ds, "df") else None
    if labels is None:
        sampler = None
    else:
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = np.array([class_weights[int(y)] for y in labels], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    # Loss
    if config.use_focal_loss:
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    # Model
    model = OCTTraigeModel(
        embed_dim=config.embed_dim,
        num_classes=2,
        oct_num_slices=config.oct_frames,
        dropout=config.dropout,
        num_centers=num_centers,
        memory_capacity=config.memory_capacity,
        alpha_cf=config.alpha_cf,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_auc = -1.0
    best_path = Path(config.checkpoint_dir) / "best_model.pt"
    last_path = Path(config.checkpoint_dir) / "last_model.pt"
    history = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            config=config,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_auc": train_metrics["auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "train_f1": train_metrics["f1"],
            "train_ppv": train_metrics["ppv"],
            "train_npv": train_metrics["npv"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "train_balanced_acc": train_metrics["balanced_acc"],
            "train_mcc": train_metrics["mcc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
            "val_ppv": val_metrics["ppv"],
            "val_npv": val_metrics["npv"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_mcc": val_metrics["mcc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_auc={train_metrics['auc']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} train_bal_acc={train_metrics['balanced_acc']:.4f} train_mcc={train_metrics['mcc']:.4f} "
            f"train_ppv={train_metrics['ppv']:.4f} train_npv={train_metrics['npv']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_bal_acc={val_metrics['balanced_acc']:.4f} val_mcc={val_metrics['mcc']:.4f} "
            f"val_ppv={val_metrics['ppv']:.4f} val_npv={val_metrics['npv']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_auc": best_auc,
                    "best_epoch_metrics": epoch_record,
                    "config": config.__dict__,
                },
                best_path,
            )
            print(f"  [OCT_traige] Saved best model to {best_path} (AUC={best_auc:.4f})")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": config.num_epochs,
            "best_auc": best_auc,
            "config": config.__dict__,
            "history": history,
        },
        last_path,
    )

    history_df = pd.DataFrame(history)
    history_csv = Path(config.log_dir) / f"metrics_history_{timestamp}.csv"
    history_json = Path(config.log_dir) / f"metrics_history_{timestamp}.json"
    history_df.to_csv(history_csv, index=False, encoding="utf-8")
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    best_row = max(history, key=lambda x: x["val_auc"]) if history else {}
    print(f"[OCT_traige] Best val AUC = {best_auc:.4f}")
    print(f"[OCT_traige] Best epoch metrics: {best_row}")
    print(f"[OCT_traige] Saved last model to {last_path}")
    print(f"[OCT_traige] Saved metrics csv to {history_csv}")
    print(f"[OCT_traige] Saved metrics json to {history_json}")


if __name__ == "__main__":
    main()

