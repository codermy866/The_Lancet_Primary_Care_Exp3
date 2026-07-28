#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset_oct_only import _extract_center_id_from_oct_id


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap_auc_ci(y_true: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y_true)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(float(roc_auc_score(yb, score[idx])))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _metrics_for_group(
    *,
    cohort: str,
    centre_id: str,
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    pred = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = int(len(y_true))
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, n)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = float(np.nanmean([sensitivity, specificity]))
    youden_j = sensitivity + specificity - 1.0 if np.isfinite(sensitivity + specificity) else float("nan")
    lr_positive = _safe_div(sensitivity, 1.0 - specificity) if np.isfinite(specificity) else float("nan")
    lr_negative = _safe_div(1.0 - sensitivity, specificity) if np.isfinite(sensitivity) else float("nan")

    auroc = float("nan")
    pr_auc = float("nan")
    auc_ci_low = float("nan")
    auc_ci_high = float("nan")
    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, score))
        pr_auc = float(average_precision_score(y_true, score))
        auc_ci_low, auc_ci_high = _bootstrap_auc_ci(y_true, score, n_boot=n_boot, seed=seed)

    sens_low, sens_high = _wilson_ci(int(tp), int(tp + fn))
    spec_low, spec_high = _wilson_ci(int(tn), int(tn + fp))
    ppv_low, ppv_high = _wilson_ci(int(tp), int(tp + fp))
    npv_low, npv_high = _wilson_ci(int(tn), int(tn + fn))
    acc_low, acc_high = _wilson_ci(int(tp + tn), n)

    return {
        "cohort": cohort,
        "centre_id": str(centre_id),
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": _safe_div(n_pos, n),
        "threshold": float(threshold),
        "auroc": auroc,
        "auroc_ci_low": auc_ci_low,
        "auroc_ci_high": auc_ci_high,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "sensitivity_ci_low": sens_low,
        "sensitivity_ci_high": sens_high,
        "specificity": specificity,
        "specificity_ci_low": spec_low,
        "specificity_ci_high": spec_high,
        "ppv": ppv,
        "ppv_ci_low": ppv_low,
        "ppv_ci_high": ppv_high,
        "npv": npv,
        "npv_ci_low": npv_low,
        "npv_ci_high": npv_high,
        "f1": f1,
        "accuracy": accuracy,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": youden_j,
        "lr_positive": lr_positive,
        "lr_negative": lr_negative,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _fmt(x: float) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def _fmt_ci(row, key: str) -> str:
    return f"{_fmt(row[key])} ({_fmt(row[key + '_ci_low'])}-{_fmt(row[key + '_ci_high'])})"


def _write_tables(df: pd.DataFrame, stem: Path, score_col: str, threshold: float) -> None:
    display = df.copy()
    display["AUROC (95% CI)"] = display.apply(lambda r: _fmt_ci(r, "auroc"), axis=1)
    display["Sensitivity (95% CI)"] = display.apply(lambda r: _fmt_ci(r, "sensitivity"), axis=1)
    display["Specificity (95% CI)"] = display.apply(lambda r: _fmt_ci(r, "specificity"), axis=1)
    display["PPV (95% CI)"] = display.apply(lambda r: _fmt_ci(r, "ppv"), axis=1)
    display["NPV (95% CI)"] = display.apply(lambda r: _fmt_ci(r, "npv"), axis=1)
    display["Accuracy (95% CI)"] = display.apply(lambda r: _fmt_ci(r, "accuracy"), axis=1)
    display["PR-AUC"] = display["pr_auc"].map(_fmt)
    display["F1"] = display["f1"].map(_fmt)
    display["Balanced accuracy"] = display["balanced_accuracy"].map(_fmt)
    display["Youden J"] = display["youden_j"].map(_fmt)
    display["LR+"] = display["lr_positive"].map(_fmt)
    display["LR-"] = display["lr_negative"].map(_fmt)
    display["n"] = display["n"].astype(int)
    display["Positive"] = display["n_positive"].astype(int)
    display["Negative"] = display["n_negative"].astype(int)

    cols = [
        "centre_id",
        "n",
        "Positive",
        "Negative",
        "AUROC (95% CI)",
        "PR-AUC",
        "Sensitivity (95% CI)",
        "Specificity (95% CI)",
        "PPV (95% CI)",
        "NPV (95% CI)",
        "F1",
        "Accuracy (95% CI)",
        "Balanced accuracy",
        "Youden J",
        "LR+",
        "LR-",
    ]
    out = display[cols].rename(columns={"centre_id": "Centre"})

    md_lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in out.columns) + " |")
    stem.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def esc(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%")

    tex_lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\hline",
        r"Centre & n & Pos & Neg & AUROC (95\% CI) & PR-AUC & Sens (95\% CI) & Spec (95\% CI) & PPV (95\% CI) & NPV (95\% CI) & F1 \\",
        r"\hline",
    ]
    for _, row in out.iterrows():
        tex_lines.append(
            f"{esc(row['Centre'])} & {row['n']} & {row['Positive']} & {row['Negative']} & "
            f"{esc(row['AUROC (95% CI)'])} & {row['PR-AUC']} & {esc(row['Sensitivity (95% CI)'])} & "
            f"{esc(row['Specificity (95% CI)'])} & {esc(row['PPV (95% CI)'])} & "
            f"{esc(row['NPV (95% CI)'])} & {row['F1']} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    stem.with_suffix(".tex").write_text("\n".join(tex_lines), encoding="utf-8")

    notes = {
        "score_col": score_col,
        "threshold": threshold,
        "notes": [
            "AUROC confidence intervals use patient-level percentile bootstrap.",
            "Sensitivity, specificity, PPV, NPV and accuracy confidence intervals use Wilson intervals.",
            "Threshold-dependent metrics use the fixed threshold supplied to this script; no threshold was optimized on the external labels.",
        ],
    }
    stem.with_suffix(".notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lancet-style overall/per-centre metrics from prediction scores.")
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--score_col", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--out_dir", default="logs/lancet_center_metrics")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--cohort", default="external_test")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv)
    if "oct_id" not in df.columns or "label" not in df.columns:
        raise ValueError("prediction CSV must contain oct_id and label")
    if args.score_col not in df.columns:
        raise ValueError(f"score column not found: {args.score_col}")
    if "centre_id" not in df.columns and "center_id" not in df.columns:
        df["centre_id"] = df["oct_id"].map(_extract_center_id_from_oct_id)
    elif "center_id" in df.columns:
        df["centre_id"] = df["center_id"].astype(str)
    else:
        df["centre_id"] = df["centre_id"].astype(str)

    y = df["label"].astype(int).to_numpy()
    score = df[args.score_col].astype(float).to_numpy()
    rows = [
        _metrics_for_group(
            cohort=args.cohort,
            centre_id="Overall",
            y_true=y,
            score=score,
            threshold=args.threshold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    ]
    for i, (centre, sub) in enumerate(df.groupby("centre_id", sort=True), start=1):
        rows.append(
            _metrics_for_group(
                cohort=args.cohort,
                centre_id=str(centre),
                y_true=sub["label"].astype(int).to_numpy(),
                score=sub[args.score_col].astype(float).to_numpy(),
                threshold=args.threshold,
                n_boot=args.n_boot,
                seed=args.seed + i * 101,
            )
        )

    metrics = pd.DataFrame(rows)
    prefix = args.prefix or args.score_col
    stem = out_dir / prefix
    metrics.to_csv(stem.with_suffix(".csv"), index=False)
    metrics.to_json(stem.with_suffix(".json"), orient="records", force_ascii=False, indent=2)
    _write_tables(metrics, stem, args.score_col, args.threshold)

    print(f"[saved] {stem.with_suffix('.csv')}")
    print(f"[saved] {stem.with_suffix('.md')}")
    print(f"[saved] {stem.with_suffix('.tex')}")
    print(metrics[["centre_id", "n", "n_positive", "auroc", "pr_auc", "sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
