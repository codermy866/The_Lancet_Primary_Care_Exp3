from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _collect_images_from_dir(oct_dir: Path) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(oct_dir.glob(ext)))
    return [str(p) for p in files]


def _load_5centers_hospital_map(medical_info_xlsx: Path) -> Dict[str, str]:
    df = pd.read_excel(medical_info_xlsx, sheet_name="MedicalInfo")
    if "OCT图像Id" not in df.columns or "医院" not in df.columns:
        return {}
    mapping = (
        df[["OCT图像Id", "医院"]]
        .dropna(subset=["OCT图像Id"])
        .assign(OCT图像Id=lambda x: x["OCT图像Id"].astype(str).str.strip())
        .assign(医院=lambda x: x["医院"].astype(str).str.strip())
    )
    return dict(zip(mapping["OCT图像Id"], mapping["医院"]))


def _build_internal_rows(data_root_5c: Path) -> pd.DataFrame:
    train_csv = data_root_5c / "train_labels.csv"
    test_csv = data_root_5c / "test_labels.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("5centers_multi 缺少 train_labels.csv 或 test_labels.csv")

    df_train = pd.read_csv(train_csv, encoding="utf-8-sig")
    df_test = pd.read_csv(test_csv, encoding="utf-8-sig")
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    if "OCT" not in df_all.columns or "label" not in df_all.columns:
        raise ValueError("5centers 标签文件必须包含 `OCT` 和 `label` 列")

    hospital_map = _load_5centers_hospital_map(data_root_5c / "3000_num.xlsx")
    rows: List[Dict] = []
    missing_images = 0
    missing_label = 0

    for _, r in df_all.iterrows():
        oct_id = str(r["OCT"]).strip()
        if not oct_id or oct_id.lower() == "nan":
            continue
        try:
            label = int(float(r["label"]))
        except Exception:
            missing_label += 1
            continue
        split_dir = "train" if (data_root_5c / "train" / "oct" / oct_id).exists() else "test"
        oct_dir = data_root_5c / split_dir / "oct" / oct_id
        paths = _collect_images_from_dir(oct_dir)
        if not paths:
            missing_images += 1
            continue

        center_name = hospital_map.get(oct_id, "5c_unknown_center")
        rows.append(
            {
                "oct_id": oct_id,
                "label": label,
                "center_id": center_name,
                "oct_paths": ";".join(paths),
                "source": "5centers_multi",
            }
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["oct_id"], keep="first").reset_index(drop=True)
    print(f"[internal] usable={len(out)} missing_images={missing_images} missing_label={missing_label}")
    return out


def _build_external_rows(data_root_10c: Path) -> pd.DataFrame:
    centers = [
        ("AnYang", "AnYang_dataset.csv"),
        ("Hua_Xi", "HuaXi_dataset.csv"),
        ("liaoning", "LiaoNing_dataset.csv"),
        ("XiangYa", "XiangYa_dataset.csv"),
        ("ZhengDaSanFu", "ZhengDaSanFu_dataset.csv"),
    ]
    rows: List[Dict] = []
    missing_images = 0
    missing_label = 0

    for center, csv_name in centers:
        csv_path = data_root_10c / csv_name
        center_dir = data_root_10c / center
        if not csv_path.exists():
            raise FileNotFoundError(f"缺少 CSV: {csv_path}")
        if not center_dir.exists():
            raise FileNotFoundError(f"缺少中心目录: {center_dir}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "OCT_ID" not in df.columns or "Final_Label" not in df.columns:
            raise ValueError(f"{csv_path} 缺少 OCT_ID/Final_Label 列")

        for _, r in df.iterrows():
            oct_id = str(r["OCT_ID"]).strip()
            if not oct_id or oct_id.lower() == "nan":
                continue
            v = r["Final_Label"]
            if pd.isna(v) or str(v).strip() == "":
                missing_label += 1
                continue
            try:
                label = int(float(v))
            except Exception:
                missing_label += 1
                continue
            oct_dir = center_dir / oct_id
            paths = _collect_images_from_dir(oct_dir)
            if not paths:
                missing_images += 1
                continue
            rows.append(
                {
                    "oct_id": oct_id,
                    "label": label,
                    "center_id": center,
                    "oct_paths": ";".join(paths),
                    "source": "10center_datas",
                }
            )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["oct_id"], keep="first").reset_index(drop=True)
    print(f"[external] usable={len(out)} missing_images={missing_images} missing_label={missing_label}")
    return out


def _stratified_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["label"].nunique() < 2:
        raise ValueError("内部数据只有单一类别，无法分层划分 train/val")
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_ratio,
        random_state=seed,
        stratify=df["label"].values,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare LOC-5out dataset for OCT_traige")
    parser.add_argument("--data_root_5c", type=str, required=True)
    parser.add_argument("--data_root_10c", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "internal_train" / "train" / "oct").mkdir(parents=True, exist_ok=True)
    (out_root / "internal_train" / "val" / "oct").mkdir(parents=True, exist_ok=True)

    internal_df = _build_internal_rows(Path(args.data_root_5c))
    external_df = _build_external_rows(Path(args.data_root_10c))
    train_df, val_df = _stratified_split(internal_df, val_ratio=args.val_ratio, seed=args.seed)

    train_csv = out_root / "train_labels.csv"
    val_csv = out_root / "val_labels.csv"
    ext_csv = out_root / "external_test_labels.csv"
    report_csv = out_root / "dataset_summary.csv"

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    external_df.to_csv(ext_csv, index=False, encoding="utf-8")

    summary = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            external_df.assign(split="external"),
        ],
        ignore_index=True,
    )
    stats = (
        summary.groupby(["split", "center_id", "label"]).size().reset_index(name="count").sort_values(["split", "center_id", "label"])
    )
    stats.to_csv(report_csv, index=False, encoding="utf-8")

    print(f"[done] train={len(train_df)} val={len(val_df)} external={len(external_df)}")
    print(f"[saved] {train_csv}")
    print(f"[saved] {val_csv}")
    print(f"[saved] {ext_csv}")
    print(f"[saved] {report_csv}")


if __name__ == "__main__":
    main()

