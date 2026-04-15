from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _fmt(x) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        if pd.isna(x):
            return "NA"
    except (TypeError, ValueError):
        pass
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:.3f}"
    except (TypeError, ValueError):
        return str(x)


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


# English site labels for manuscript tables (edit or extend as needed).
_SITE_LABEL_EN: dict[str, str] = {
    # Internal development sites (Chinese names in CSV)
    "十堰市人民医院": "Shiyan People's Hospital",
    "恩施州中心医院": "Enshi Prefecture Central Hospital",
    "武大人民医院": "Renmin Hospital of Wuhan University",
    "荆州市第一人民医院": "Jingzhou First People's Hospital",
    "襄阳市中心医院": "Xiangyang Central Hospital",
    "5c_unknown_center": "Unmapped site (excluded from per-site tables)",
    # External held-out sites (folder / table ids)
    "AnYang": "Anyang",
    "Hua_Xi": "Huaxi",
    "HuaXi": "Huaxi",
    "liaoning": "Liaoning",
    "ZhengDaSanFu": "Zhengda Sanfu",
    "XiangYa": "Xiangya",
}


def _site_label_en(site_id: str) -> str:
    s = str(site_id).strip()
    if s in _SITE_LABEL_EN:
        return _latex_escape(_SITE_LABEL_EN[s])
    return _latex_escape(s)


def _load_best_epoch_row(history_path: Path) -> tuple[int, dict]:
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    if not isinstance(hist, list) or not hist:
        raise ValueError(f"metrics_history must be a non-empty list: {history_path}")
    best = max(hist, key=lambda x: float(x.get("val_auc", 0.0)))
    return int(best["epoch"]), best


def _count_rows(csv_path: Path) -> int:
    return len(pd.read_csv(csv_path, encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate journal-style LaTeX tables (English): internal + external (LOC-5out)."
    )
    parser.add_argument("--overall_json", type=str, required=True, help="external_metrics_loc5out.json")
    parser.add_argument("--per_center_csv", type=str, required=True)
    parser.add_argument("--metrics_history_json", type=str, default="logs/metrics_history_20260402_223023.json")
    parser.add_argument("--train_csv", type=str, default="", help="Override default data_root/train_labels.csv")
    parser.add_argument("--val_csv", type=str, default="", help="Override default data_root/val_labels.csv")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data2/hmy/VLM_Caus_Rm_Mics/data/loc5out_10centers_oct",
        help="Data root for default train/val CSV paths and sample counts",
    )
    parser.add_argument(
        "--internal_train_per_center_csv",
        type=str,
        default="logs/internal_train_per_center_metrics_loc5out.csv",
        help="From eval_internal_oct_traige.py: per-site metrics on training split",
    )
    parser.add_argument(
        "--internal_val_per_center_csv",
        type=str,
        default="logs/internal_val_per_center_metrics_loc5out.csv",
        help="From eval_internal_oct_traige.py: per-site metrics on validation split",
    )
    parser.add_argument("--out_tex", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    overall_json = Path(args.overall_json)
    if not overall_json.is_absolute():
        overall_json = root / overall_json
    per_center_csv = Path(args.per_center_csv)
    if not per_center_csv.is_absolute():
        per_center_csv = root / per_center_csv
    history_path = Path(args.metrics_history_json)
    if not history_path.is_absolute():
        history_path = root / history_path
    out_tex = Path(args.out_tex)
    if not out_tex.is_absolute():
        out_tex = root / out_tex

    data_root = Path(args.data_root)
    if args.train_csv:
        train_csv = Path(args.train_csv)
        train_csv = train_csv if train_csv.is_absolute() else data_root / train_csv.name
    else:
        train_csv = data_root / "train_labels.csv"

    if args.val_csv:
        val_csv = Path(args.val_csv)
        val_csv = val_csv if val_csv.is_absolute() else data_root / val_csv.name
    else:
        val_csv = data_root / "val_labels.csv"

    n_train = _count_rows(train_csv) if train_csv.exists() else 788
    n_val = _count_rows(val_csv) if val_csv.exists() else 197

    best_epoch, row = _load_best_epoch_row(history_path)

    with open(overall_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    overall = payload.get("overall", {})
    n_ext = int(payload.get("num_samples", 0))
    per_center = pd.read_csv(per_center_csv, encoding="utf-8")

    int_train_pc = Path(args.internal_train_per_center_csv)
    int_val_pc = Path(args.internal_val_per_center_csv)
    if not int_train_pc.is_absolute():
        int_train_pc = root / int_train_pc
    if not int_val_pc.is_absolute():
        int_val_pc = root / int_val_pc

    needed = ["auc", "f1", "ppv", "npv", "sensitivity", "specificity"]
    for k in needed:
        if k not in overall:
            raise ValueError(f"overall metrics missing key: {k}")

    lines: list[str] = []
    lines.append("% Auto-generated by generate_lancet_latex_tables.py (English labels for manuscript use).")
    lines.append("% Table 1 summary: epoch-level metrics from training logs (epoch with highest validation AUROC).")
    lines.append("% Per-site internal rows: inference with best_model.pt in eval mode (eval_internal_oct_traige.py).")
    lines.append("% Binary metrics: class prediction via argmax unless otherwise specified in the main text.")
    lines.append("% NA = not applicable / undefined (e.g. AUROC when only one class in that stratum).")
    lines.append("% Site names: mapped to English where available; extend _SITE_LABEL_EN in the script if needed.")
    lines.append("")

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Development cohort (five participating sites; leave-out external-test design). "
        "Summary metrics at epoch "
        + str(best_epoch)
        + " (highest validation AUROC; same checkpoint as external evaluation).}"
    )
    lines.append("\\label{tab:internal_dev_loc5out}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("Data split & AUROC & F1 score & PPV & NPV & Sensitivity & Specificity \\\\")
    lines.append("\\hline")
    lines.append(
        f"Training set (n={n_train}) & {_fmt(float(row['train_auc']))} & {_fmt(float(row['train_f1']))} & "
        f"{_fmt(float(row['train_ppv']))} & {_fmt(float(row['train_npv']))} & "
        f"{_fmt(float(row['train_sensitivity']))} & {_fmt(float(row['train_specificity']))} \\\\"
    )
    lines.append(
        f"Validation set (n={n_val}) & {_fmt(float(row['val_auc']))} & {_fmt(float(row['val_f1']))} & "
        f"{_fmt(float(row['val_ppv']))} & {_fmt(float(row['val_npv']))} & "
        f"{_fmt(float(row['val_sensitivity']))} & {_fmt(float(row['val_specificity']))} \\\\"
    )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    if int_train_pc.exists():
        df_it = pd.read_csv(int_train_pc, encoding="utf-8")
        df_it = df_it[df_it["center_id"].astype(str) != "5c_unknown_center"]
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(
            "\\caption{Development cohort --- training partition: performance by participating site. "
            "Metrics from \\texttt{best\\_model.pt} under evaluation mode (no dropout).}"
        )
        lines.append("\\label{tab:internal_train_per_center_loc5out}")
        lines.append("\\begin{tabular}{lccccccc}")
        lines.append("\\hline")
        lines.append("Site & n & AUROC & F1 score & PPV & NPV & Sensitivity & Specificity \\\\")
        lines.append("\\hline")
        for _, r in df_it.sort_values("center_id").iterrows():
            cid = _site_label_en(r["center_id"])
            lines.append(
                f"{cid} & {int(r['n'])} & {_fmt(float(r['auc']))} & {_fmt(float(r['f1']))} & "
                f"{_fmt(float(r['ppv']))} & {_fmt(float(r['npv']))} & {_fmt(float(r['sensitivity']))} & "
                f"{_fmt(float(r['specificity']))} \\\\"
            )
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    else:
        lines.append(
            "% [missing] internal train per-site CSV --- run: python training/eval_internal_oct_traige.py"
        )
        lines.append("")

    if int_val_pc.exists():
        df_iv = pd.read_csv(int_val_pc, encoding="utf-8")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(
            "\\caption{Development cohort --- validation partition: performance by participating site. "
            "Metrics from \\texttt{best\\_model.pt} under evaluation mode (no dropout).}"
        )
        lines.append("\\label{tab:internal_val_per_center_loc5out}")
        lines.append("\\begin{tabular}{lccccccc}")
        lines.append("\\hline")
        lines.append("Site & n & AUROC & F1 score & PPV & NPV & Sensitivity & Specificity \\\\")
        lines.append("\\hline")
        for _, r in df_iv.sort_values("center_id").iterrows():
            cid = _site_label_en(r["center_id"])
            lines.append(
                f"{cid} & {int(r['n'])} & {_fmt(float(r['auc']))} & {_fmt(float(r['f1']))} & "
                f"{_fmt(float(r['ppv']))} & {_fmt(float(r['npv']))} & {_fmt(float(r['sensitivity']))} & "
                f"{_fmt(float(r['specificity']))} \\\\"
            )
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    else:
        lines.append(
            "% [missing] internal validation per-site CSV --- run: python training/eval_internal_oct_traige.py"
        )
        lines.append("")

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{External test cohort (held-out sites; independent of model selection). Overall performance.}"
    )
    lines.append("\\label{tab:external_overall_loc5out}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("Cohort & AUROC & F1 score & PPV & NPV & Sensitivity & Specificity \\\\")
    lines.append("\\hline")
    lines.append(
        f"External test set (n={n_ext}) & {_fmt(overall['auc'])} & {_fmt(overall['f1'])} & {_fmt(overall['ppv'])} & "
        f"{_fmt(overall['npv'])} & {_fmt(overall['sensitivity'])} & {_fmt(overall['specificity'])} \\\\"
    )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{External test cohort: performance by participating site.}")
    lines.append("\\label{tab:external_per_center_loc5out}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\hline")
    lines.append("Site & n & AUROC & F1 score & PPV & NPV & Sensitivity & Specificity \\\\")
    lines.append("\\hline")
    for _, r in per_center.sort_values("center_id").iterrows():
        cid = _site_label_en(r["center_id"])
        lines.append(
            f"{cid} & {int(r['n'])} & {_fmt(float(r['auc']))} & {_fmt(float(r['f1']))} & "
            f"{_fmt(float(r['ppv']))} & {_fmt(float(r['npv']))} & {_fmt(float(r['sensitivity']))} & "
            f"{_fmt(float(r['specificity']))} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_tex}")


if __name__ == "__main__":
    main()
