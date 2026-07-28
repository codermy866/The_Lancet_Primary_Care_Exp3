#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset_oct_only import _natural_sort_key


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="gbk")
    if "oct_id" not in df.columns and "OCT" in df.columns:
        df = df.rename(columns={"OCT": "oct_id"})
    if "oct_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain oct_id/OCT and label columns")
    return df


def _case_paths(data_root: Path, split: str, oct_id: str) -> list[str]:
    if split == "external":
        base = data_root / "external_validation" / "oct"
    else:
        base = data_root / "internal_train" / split / "oct"
    files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        files.extend((base / str(oct_id)).glob(ext))
    return [str(p) for p in sorted(files, key=_natural_sort_key)]


def _sample_paths(paths: list[str], n_frames: int) -> list[str]:
    if not paths:
        return []
    if len(paths) >= n_frames:
        idx = np.linspace(0, len(paths) - 1, n_frames, dtype=int)
        return [paths[i] for i in idx]
    out = list(paths)
    while len(out) < n_frames:
        out.extend(paths)
    return out[:n_frames]


class FrameDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform, n_frames: int, path_col: str):
        self.items: list[tuple[int, str]] = []
        for case_idx, paths in enumerate(rows[path_col].tolist()):
            for p in _sample_paths(paths, n_frames):
                self.items.append((case_idx, p))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        case_idx, path = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
        except Exception:
            x = torch.zeros(3, 224, 224)
        return case_idx, x


def _build_backbone(name: str, device: torch.device) -> tuple[nn.Module, transforms.Compose, int]:
    name = name.lower()
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        transform = weights.transforms()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        net = models.efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        transform = weights.transforms()
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        net = models.convnext_tiny(weights=weights)
        feat_dim = net.classifier[2].in_features
        net.classifier = nn.Identity()
        transform = weights.transforms()
    elif name in {"vit_l_16", "vit_l_16_imagenet"}:
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        net = models.vit_l_16(weights=weights)
        feat_dim = net.heads.head.in_features
        net.heads = nn.Identity()
        transform = weights.transforms()
    elif name == "vit_l_16_swag_linear":
        weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        net = models.vit_l_16(weights=weights)
        feat_dim = net.heads.head.in_features
        net.heads = nn.Identity()
        transform = weights.transforms()
    elif name == "vit_l_16_swag_e2e":
        weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        net = models.vit_l_16(weights=weights)
        feat_dim = net.heads.head.in_features
        net.heads = nn.Identity()
        transform = weights.transforms()
    elif name in {"dinov2_vitl14", "dinov2_vitl14_reg"}:
        # DINOv2 hub models return the CLS embedding directly.
        net = torch.hub.load("facebookresearch/dinov2", name, pretrained=True, trust_repo=True)
        feat_dim = 1024
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        raise ValueError(f"unknown backbone: {name}")
    net = net.to(device).eval()
    return net, transform, feat_dim


@torch.no_grad()
def _extract_features(
    *,
    rows: pd.DataFrame,
    backbone: nn.Module,
    transform,
    feat_dim: int,
    n_frames: int,
    path_col: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    ds = FrameDataset(rows, transform=transform, n_frames=n_frames, path_col=path_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    sums = np.zeros((len(rows), feat_dim), dtype=np.float64)
    sums_sq = np.zeros((len(rows), feat_dim), dtype=np.float64)
    maxs = np.full((len(rows), feat_dim), -np.inf, dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.float64)

    for case_idx, x in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        feat = backbone(x)
        if isinstance(feat, dict):
            feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        feat_np = feat.detach().cpu().numpy().astype(np.float64)
        case_np = case_idx.numpy()
        for i, c in enumerate(case_np):
            sums[c] += feat_np[i]
            sums_sq[c] += feat_np[i] ** 2
            maxs[c] = np.maximum(maxs[c], feat_np[i])
            counts[c] += 1

    counts = np.maximum(counts, 1.0)[:, None]
    mean = sums / counts
    var = np.maximum((sums_sq / counts) - mean**2, 0.0)
    std = np.sqrt(var)
    maxs[~np.isfinite(maxs)] = 0.0
    return np.concatenate([mean, std, maxs], axis=1).astype(np.float32)


def _clinical_matrix(train_df: pd.DataFrame, val_df: pd.DataFrame, ext_df: pd.DataFrame):
    cols = [c for c in ["AGE", "HPV清洗", "TCT清洗"] if c in train_df.columns]
    if not cols:
        return None, None, None, []
    train = train_df[cols].copy()
    val = val_df[cols].copy()
    ext = ext_df[cols].copy()
    num_cols = [c for c in cols if c == "AGE"]
    cat_cols = [c for c in cols if c != "AGE"]
    for df in (train, val, ext):
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in cat_cols:
            df[c] = df[c].astype(str).replace({"nan": np.nan})
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]),
                cat_cols,
            ),
        ]
    )
    x_train = pre.fit_transform(train)
    x_val = pre.transform(val)
    x_ext = pre.transform(ext)
    return x_train.toarray() if hasattr(x_train, "toarray") else x_train, x_val.toarray() if hasattr(x_val, "toarray") else x_val, x_ext.toarray() if hasattr(x_ext, "toarray") else x_ext, cols


def _fit_candidates(x_train, y_train, x_val, y_val, x_ext, y_ext) -> tuple[list[dict], dict, np.ndarray]:
    candidates = [
        ("ridge_a0.1", RidgeClassifier(alpha=0.1, class_weight="balanced")),
        ("ridge_a1", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ("ridge_a3", RidgeClassifier(alpha=3.0, class_weight="balanced")),
        ("ridge_a10", RidgeClassifier(alpha=10.0, class_weight="balanced")),
        ("ridge_a30", RidgeClassifier(alpha=30.0, class_weight="balanced")),
    ]
    rows: list[dict] = []
    best_name = ""
    best_val_auc = -1.0
    best_prob_ext = None
    for name, clf in candidates:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(x_train, y_train)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            p_val = pipe.predict_proba(x_val)[:, 1]
            p_ext = pipe.predict_proba(x_ext)[:, 1]
        else:
            p_val = pipe.decision_function(x_val)
            p_ext = pipe.decision_function(x_ext)
        val_auc = float(roc_auc_score(y_val, p_val))
        ext_auc = float(roc_auc_score(y_ext, p_ext))
        rows.append(
            {
                "model": name,
                "val_auc": val_auc,
                "val_pr_auc": float(average_precision_score(y_val, p_val)),
                "external_auc": ext_auc,
                "external_pr_auc": float(average_precision_score(y_ext, p_ext)),
            }
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_name = name
            best_prob_ext = np.asarray(p_ext, dtype=float)
    selected = next(r for r in rows if r["model"] == best_name)
    return rows, selected, best_prob_ext


def _prepare_rows(data_root: Path, csv_name: str, split: str) -> pd.DataFrame:
    df = _read_csv(data_root / csv_name)
    df = df.copy()
    if "oct_paths" not in df.columns:
        df["oct_paths"] = df["oct_id"].astype(str).map(lambda x: _case_paths(data_root, split, x))
    else:
        df["oct_paths"] = df["oct_paths"].fillna("").astype(str).map(lambda x: [p for p in x.split(";") if p])
    if "col_paths" not in df.columns:
        if "ID" in df.columns:
            id_col = "ID"
        elif "patient_id" in df.columns:
            id_col = "patient_id"
        else:
            id_col = ""
        if id_col:
            if split == "external":
                col_base = data_root / "external_validation" / "col"
            else:
                col_base = data_root / "internal_train" / split / "col"

            def collect_col(pid):
                folder = col_base / str(pid)
                files: list[Path] = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
                    files.extend(folder.glob(ext))
                return [str(p) for p in sorted(files, key=_natural_sort_key)]

            df["col_paths"] = df[id_col].astype(str).map(collect_col)
        else:
            df["col_paths"] = [[] for _ in range(len(df))]
    else:
        df["col_paths"] = df["col_paths"].fillna("").astype(str).map(lambda x: [p for p in x.split(";") if p])
    return df


def main():
    parser = argparse.ArgumentParser(description="Frozen ImageNet feature baseline for OCT triage")
    parser.add_argument("--data_root", default="/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out")
    parser.add_argument("--backbones", nargs="+", default=["efficientnet_b0", "resnet50"])
    parser.add_argument("--n_frames", type=int, default=24)
    parser.add_argument("--n_col_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--include_clinical", type=int, default=1)
    parser.add_argument("--include_col", type=int, default=0)
    parser.add_argument("--out_dir", default="logs/frozen_feature_baseline")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = _prepare_rows(data_root, "train_labels.csv", "train")
    val_df = _prepare_rows(data_root, "val_labels.csv", "val")
    ext_df = _prepare_rows(data_root, "external_test_labels.csv", "external")
    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_ext = ext_df["label"].astype(int).to_numpy()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clinical = _clinical_matrix(train_df, val_df, ext_df) if int(args.include_clinical) else (None, None, None, [])
    clin_train, clin_val, clin_ext, clinical_cols = clinical

    all_results: list[dict] = []
    pred_df = ext_df[["oct_id", "label"]].copy()
    if "center_id" in ext_df.columns:
        pred_df["center_id"] = ext_df["center_id"].astype(str)

    for backbone_name in args.backbones:
        backbone, transform, feat_dim = _build_backbone(backbone_name, device)
        feats = {}
        for split_name, df in [("train", train_df), ("val", val_df), ("external", ext_df)]:
            cache_path = cache_dir / f"{backbone_name}_{split_name}_oct_frames{args.n_frames}.npy"
            if cache_path.exists():
                feats[f"oct_{split_name}"] = np.load(cache_path)
            else:
                arr = _extract_features(
                    rows=df,
                    backbone=backbone,
                    transform=transform,
                    feat_dim=feat_dim,
                    n_frames=int(args.n_frames),
                    path_col="oct_paths",
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    device=device,
                )
                np.save(cache_path, arr)
                feats[f"oct_{split_name}"] = arr
            if int(args.include_col):
                col_cache_path = cache_dir / f"{backbone_name}_{split_name}_col_frames{args.n_col_images}.npy"
                if col_cache_path.exists():
                    feats[f"col_{split_name}"] = np.load(col_cache_path)
                else:
                    arr = _extract_features(
                        rows=df,
                        backbone=backbone,
                        transform=transform,
                        feat_dim=feat_dim,
                        n_frames=int(args.n_col_images),
                        path_col="col_paths",
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        device=device,
                    )
                    np.save(col_cache_path, arr)
                    feats[f"col_{split_name}"] = arr

        variants = {
            "oct_only": (feats["oct_train"], feats["oct_val"], feats["oct_external"]),
        }
        if int(args.include_col):
            variants["col_only"] = (feats["col_train"], feats["col_val"], feats["col_external"])
            variants["oct_col"] = (
                np.concatenate([feats["oct_train"], feats["col_train"]], axis=1),
                np.concatenate([feats["oct_val"], feats["col_val"]], axis=1),
                np.concatenate([feats["oct_external"], feats["col_external"]], axis=1),
            )
        if clin_train is not None:
            variants["oct_clinical"] = (
                np.concatenate([feats["oct_train"], clin_train], axis=1),
                np.concatenate([feats["oct_val"], clin_val], axis=1),
                np.concatenate([feats["oct_external"], clin_ext], axis=1),
            )
            if int(args.include_col):
                variants["col_clinical"] = (
                    np.concatenate([feats["col_train"], clin_train], axis=1),
                    np.concatenate([feats["col_val"], clin_val], axis=1),
                    np.concatenate([feats["col_external"], clin_ext], axis=1),
                )
                variants["oct_col_clinical"] = (
                    np.concatenate([feats["oct_train"], feats["col_train"], clin_train], axis=1),
                    np.concatenate([feats["oct_val"], feats["col_val"], clin_val], axis=1),
                    np.concatenate([feats["oct_external"], feats["col_external"], clin_ext], axis=1),
                )

        for variant, (x_train, x_val, x_ext) in variants.items():
            rows, selected, prob_ext = _fit_candidates(x_train, y_train, x_val, y_val, x_ext, y_ext)
            record = {
                "backbone": backbone_name,
                "variant": variant,
                "n_frames": int(args.n_frames),
                "clinical_cols": clinical_cols if variant == "oct_clinical" else [],
                "selected_by_internal_val": selected,
                "candidates": rows,
            }
            all_results.append(record)
            pred_df[f"prob_{backbone_name}_{variant}"] = prob_ext

    # A simple ensemble of internally selected frozen-feature variants.
    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    if prob_cols:
        pred_df["prob_frozen_mean"] = pred_df[prob_cols].mean(axis=1)
        all_results.append(
            {
                "backbone": "ensemble",
                "variant": "mean_selected_variants",
                "selected_by_internal_val": {
                    "model": "mean",
                    "external_auc": float(roc_auc_score(y_ext, pred_df["prob_frozen_mean"])),
                    "external_pr_auc": float(average_precision_score(y_ext, pred_df["prob_frozen_mean"])),
                },
                "prob_cols": prob_cols,
            }
        )

    out_json = out_dir / "frozen_feature_results.json"
    out_csv = out_dir / "external_predictions_frozen_features.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "n_train": len(train_df), "n_val": len(val_df), "n_external": len(ext_df)}, f, ensure_ascii=False, indent=2)
    pred_df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")
    for r in all_results:
        sel = r["selected_by_internal_val"]
        print(
            f"{r['backbone']}/{r['variant']}: "
            f"selected={sel.get('model')} val_auc={sel.get('val_auc', float('nan')):.4f} "
            f"external_auc={sel.get('external_auc'):.4f} external_pr={sel.get('external_pr_auc'):.4f}"
        )


if __name__ == "__main__":
    main()
