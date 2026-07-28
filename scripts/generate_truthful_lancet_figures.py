#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset_oct_only import _extract_center_id_from_oct_id

PRED_CSV = ROOT / "logs/final_external_ensemble_20260507_dinov2/external_predictions_dinov2_ensemble.csv"
LABEL_CSV = Path("/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out/external_test_labels.csv")
SCORE_COL = "old_all_plus_all_dino_variants_center_z_mean"
THRESHOLD = 0.0
OUT_DIR = ROOT / "logs/lancet_truthful_figures_20260507"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(x: float) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.2f}"


def fmt3(x: float) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.3f}"


def ci(point: float, lo: float, hi: float) -> str:
    return f"{fmt(point)} ({fmt(lo)}--{fmt(hi)})"


def ci3(point: float, lo: float, hi: float) -> str:
    return f"{fmt3(point)} ({fmt3(lo)}--{fmt3(hi)})"


def metrics(name: str, g: pd.DataFrame, section: str = "") -> dict:
    y = g["label"].astype(int).to_numpy()
    pred = (g[SCORE_COL].astype(float).to_numpy() >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = int(len(g))
    pos = int(tp + fn)
    neg = int(tn + fp)
    sens = tp / pos if pos else float("nan")
    fpb = fp / neg if neg else float("nan")
    spec = tn / neg if neg else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    sens_lo, sens_hi = wilson(int(tp), int(pos))
    fpb_lo, fpb_hi = wilson(int(fp), int(neg))
    spec_lo, spec_hi = wilson(int(tn), int(neg))
    return {
        "section": section,
        "group": name,
        "n": n,
        "positive": pos,
        "negative": neg,
        "prevalence": pos / n if n else float("nan"),
        "sensitivity": sens,
        "sensitivity_ci_low": sens_lo,
        "sensitivity_ci_high": sens_hi,
        "fp_burden": fpb,
        "fp_burden_ci_low": fpb_lo,
        "fp_burden_ci_high": fpb_hi,
        "specificity": spec,
        "specificity_ci_low": spec_lo,
        "specificity_ci_high": spec_hi,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "accuracy": acc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_data() -> pd.DataFrame:
    pred = pd.read_csv(PRED_CSV)
    lab = pd.read_csv(LABEL_CSV, encoding="utf-8-sig").rename(columns={"OCT": "oct_id"})
    df = pred[["oct_id", "label", SCORE_COL]].merge(
        lab[["oct_id", "AGE", "HPV清洗", "TCT清洗"]],
        on="oct_id",
        how="left",
    )
    df["centre_id"] = df["oct_id"].map(_extract_center_id_from_oct_id)
    df["age_group"] = pd.cut(
        pd.to_numeric(df["AGE"], errors="coerce"),
        bins=[-np.inf, 39, 49, np.inf],
        labels=["<40 years", "40--49 years", r"$\geq$50 years"],
    )

    def hpv_group(x) -> str:
        if pd.isna(x):
            return "HPV missing"
        s = str(x).strip()
        if s == "-":
            return "HPV negative/undetected"
        return "HPV positive or genotype recorded"

    def tct_group(x) -> str:
        if pd.isna(x):
            return "TCT missing"
        s = str(x).strip()
        if s.upper() == "NILM":
            return "NILM"
        if s in {"1", "1.0"}:
            return "TCT abnormal/positive"
        return f"TCT {s}"

    df["hpv_group"] = df["HPV清洗"].map(hpv_group)
    df["tct_group"] = df["TCT清洗"].map(tct_group)
    return df


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def center_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics("Overall", df, "Overall")]
    for centre, sub in df.groupby("centre_id", sort=True):
        rows.append(metrics(str(centre), sub, "Centre"))
    return pd.DataFrame(rows)


def subgroup_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics("Overall", df, "Overall")]
    for grp in ["<40 years", "40--49 years", r"$\geq$50 years"]:
        rows.append(metrics(grp, df[df["age_group"].astype(str) == grp], "Age"))
    for grp in ["HPV positive or genotype recorded", "HPV negative/undetected", "HPV missing"]:
        rows.append(metrics(grp, df[df["hpv_group"] == grp], "HPV"))
    for grp in ["NILM", "TCT abnormal/positive", "TCT missing"]:
        rows.append(metrics(grp, df[df["tct_group"] == grp], "TCT"))
    return pd.DataFrame(rows)


def write_latex(center: pd.DataFrame, subgroups: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "% Auto-generated from current external predictions.",
            "% Endpoint: current label=1 in the dataset documentation, described as CIN1+/histology-positive.",
            "% Score: old_all_plus_all_dino_variants_center_z_mean; threshold fixed at score >= 0.0.",
            "% Do not relabel these results as CIN3+ unless a CIN3+ endpoint label is provided.",
            "",
            r"\definecolor{myred}{HTML}{9E3B46}",
            r"\definecolor{myblue}{RGB}{70,60,120}",
            r"\definecolor{tablegray}{RGB}{238,238,238}",
            r"\definecolor{softgray}{RGB}{150,150,150}",
            r"\definecolor{maingray}{RGB}{80,80,80}",
            "",
            "% ==============================================================================",
            "% FIGURE 1: Centre-level validation with truthful external-set values",
            "% ==============================================================================",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\hline",
            r"Centre & Total ($n$) & Label+ & Label+ Sens. (95\% CI) & FP burden (95\% CI) & Specificity & F1 \\",
            r"\hline",
        ]
    )
    for _, r in center.iterrows():
        lines.append(
            f"{latex_escape(r.group)} & {int(r.n)} & {int(r.positive)} & "
            f"{ci3(r.sensitivity, r.sensitivity_ci_low, r.sensitivity_ci_high)} & "
            f"{ci3(r.fp_burden, r.fp_burden_ci_low, r.fp_burden_ci_high)} & "
            f"{fmt3(r.specificity)} & {fmt3(r.f1)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{\textbf{Centre-level safety and false-positive triage burden using the current external validation set.} Values are based on the locked all-model+DINOv2 centre-normalised ensemble at the fixed threshold score $\geq 0$. Label+ denotes the current binary endpoint in the dataset (documented as CIN1+/histology-positive); FP burden is $1-\mathrm{specificity}$ among label-negative cases.}",
            r"\label{fig:truthful_centre_forest}",
            r"\end{figure}",
            "",
            "% ==============================================================================",
            "% FIGURE 2: Subgroup consistency with truthful external-set values",
            "% ==============================================================================",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{llrrrrrr}",
            r"\hline",
            r"Domain & Subgroup & Total ($n$) & Label+ & Label+ Sens. (95\% CI) & FP burden (95\% CI) & Specificity & F1 \\",
            r"\hline",
        ]
    )
    for _, r in subgroups.iterrows():
        domain = "" if r.section == "Overall" else latex_escape(r.section)
        lines.append(
            f"{domain} & {latex_escape(r.group)} & {int(r.n)} & {int(r.positive)} & "
            f"{ci3(r.sensitivity, r.sensitivity_ci_low, r.sensitivity_ci_high)} & "
            f"{ci3(r.fp_burden, r.fp_burden_ci_low, r.fp_burden_ci_high)} & "
            f"{fmt3(r.specificity)} & {fmt3(r.f1)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{\textbf{Subgroup consistency of sensitivity and false-positive burden.} Subgroups are limited to variables available in the current external labels: age, HPV-cleaned field, and TCT-cleaned field. Menopausal status was not available in the current label file and is therefore not reported.}",
            r"\label{fig:truthful_subgroup_consistency}",
            r"\end{figure}",
            "",
            "% ==============================================================================",
            "% FIGURE 3: Safety--burden trade-off bubble plot with truthful centre values",
            "% Requires: \\usepackage{pgfplots}; \\pgfplotsset{compat=1.18}",
            "% ==============================================================================",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"    width=14cm, height=9cm,",
            r"    xmin=-3, xmax=40,",
            r"    ymin=0.45, ymax=1.05,",
            r"    xtick={0,10,20,30,40},",
            r"    ytick={0.50,0.60,0.70,0.80,0.90,1.00},",
            r"    grid=major, grid style={dashed, gray!35},",
            r"    xlabel={False-positive burden among label-negative cases (\%)},",
            r"    ylabel={Label-positive sensitivity},",
            r"    tick align=outside,",
            r"    clip=false",
            r"]",
        ]
    )
    for _, r in center.iterrows():
        x = 100 * r.fp_burden if np.isfinite(r.fp_burden) else float("nan")
        y = r.sensitivity
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if r.group == "Overall":
            lines.append(
                rf"\node[diamond, draw=myblue, fill=myblue!35, minimum size=12pt] at (axis cs:{x:.1f},{y:.3f}) {{}};"
            )
            lines.append(
                rf"\node[anchor=south, align=center, font=\scriptsize\bfseries, text=myblue] at (axis cs:{x:.1f},{y:.3f}) {{Overall\\($n={int(r.n)}$)}};"
            )
        else:
            mark = "rectangle"
            fill = "myred!55" if r.positive >= 10 else "white"
            draw = "myred!80" if r.positive >= 10 else "softgray"
            size = max(5.0, min(12.0, math.sqrt(float(r.n)) * 1.2))
            lines.append(
                rf"\node[{mark}, draw={draw}, fill={fill}, thick, minimum size={size:.1f}pt] at (axis cs:{x:.1f},{y:.3f}) {{}};"
            )
            lines.append(
                rf"\node[anchor=west, font=\scriptsize] at (axis cs:{x + 1.2:.1f},{y:.3f}) {{{latex_escape(r.group)} ($n={int(r.n)}$, Label+ {int(r.positive)})}};"
            )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{\textbf{Safety--burden trade-off across external centres using truthful current results.} The x-axis shows false-positive burden ($1-\mathrm{specificity}$) among label-negative cases and the y-axis shows sensitivity among label-positive cases. The open marker denotes a centre with fewer than 10 label-positive cases.}",
            r"\label{fig:truthful_safety_burden}",
            r"\end{figure}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    centre = center_rows(df)
    subgroups = subgroup_rows(df)
    centre.to_csv(OUT_DIR / "truthful_centre_metrics.csv", index=False)
    subgroups.to_csv(OUT_DIR / "truthful_subgroup_metrics.csv", index=False)
    write_latex(centre, subgroups, OUT_DIR / "truthful_lancet_figures_20260507.tex")
    print(f"[saved] {OUT_DIR / 'truthful_centre_metrics.csv'}")
    print(f"[saved] {OUT_DIR / 'truthful_subgroup_metrics.csv'}")
    print(f"[saved] {OUT_DIR / 'truthful_lancet_figures_20260507.tex'}")
    print(centre[["group", "n", "positive", "sensitivity", "fp_burden", "specificity", "f1"]].to_string(index=False))
    print(subgroups[["section", "group", "n", "positive", "sensitivity", "fp_burden", "specificity", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
