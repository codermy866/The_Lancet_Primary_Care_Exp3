"""
顶刊风格可视化（Nature / Lancet 常见规范）：
- 无衬线字体、300 DPI PNG + 矢量 PDF
- Okabe–Ito 色盲友好配色
- 自动读取 logs 下最新 metrics_history 与评估结果

用法（在 experiments/OCT_traige 目录下）:
  python scripts/publication_figures_loc5out.py
  python scripts/publication_figures_loc5out.py --out_dir logs/figures_publication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Okabe–Ito
C_TRAIN = "#0072B2"
C_VAL = "#E69F00"
C_EXT = "#009E73"
C_NEUTRAL = "#999999"
C_NEG = "#56B4E9"


def _register_noto_cjk() -> str | None:
    """注册系统 Noto Sans CJK（TTC），供中文中心名称等标签使用。"""
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    )
    for p in candidates:
        path = Path(p)
        if path.is_file():
            try:
                font_manager.fontManager.addfont(str(path))
            except (OSError, ValueError, RuntimeError):
                continue
            # matplotlib 对同一 TTC 常登记为 JP 族名，但含完整汉字字形
            return "Noto Sans CJK JP"
    return None


def apply_journal_style() -> None:
    cjk = _register_noto_cjk()
    sans = [cjk] if cjk else []
    sans.extend(
        [
            "Noto Sans CJK SC",
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "Nimbus Sans",
            "sans-serif",
        ]
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )


def save_both(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), facecolor="white")
    fig.savefig(path_base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def latest_metrics_csv(logs: Path) -> Path:
    files = sorted(logs.glob("metrics_history_*.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics_history_*.csv in {logs}")
    return files[-1]


def fig_training_multipanel(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    ep = d["epoch"].values
    best_i = int(d["val_auc"].idxmax())
    best_ep = int(d.loc[best_i, "epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    ax = axes.ravel()

    pairs = [
        ("train_loss", "val_loss", "Loss", None),
        ("train_auc", "val_auc", "AUROC", (0.45, 1.02)),
        ("train_pr_auc", "val_pr_auc", "Area under precision–recall curve", (0.0, 1.02)),
        ("train_balanced_acc", "val_balanced_acc", "Balanced accuracy", (0.35, 1.02)),
    ]
    for i, (tc, vc, ylab, ylim) in enumerate(pairs):
        ax[i].plot(ep, d[tc], color=C_TRAIN, label="Training", zorder=2)
        ax[i].plot(ep, d[vc], color=C_VAL, label="Validation", zorder=2)
        ax[i].axvline(best_ep, color=C_NEUTRAL, ls="--", lw=1, zorder=1, alpha=0.9)
        ax[i].set_ylabel(ylab)
        if ylim:
            ax[i].set_ylim(*ylim)
        if i >= 2:
            ax[i].set_xlabel("Epoch")
    axes[0, 0].legend(loc="upper right", frameon=False)
    for a in axes[0]:
        a.set_xlabel("")
    fig.suptitle(
        "Model training (development centres, LOC-5-out split)\n"
        f"Best validation AUROC at epoch {best_ep} (vertical line)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    save_both(fig, out / "pub_fig01_training_curves")


def fig_external_roc_pr_calib(pred_csv: Path, out: Path) -> None:
    df = pd.read_csv(pred_csv, encoding="utf-8")
    y = df["label"].astype(int).values
    s = df["prob_pos"].astype(float).values

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    ax0, ax1, ax2 = axes

    # ROC
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    ax0.plot(fpr, tpr, color=C_EXT, lw=2, label=f"AUROC = {roc_auc:.3f}")
    ax0.plot([0, 1], [0, 1], color=C_NEUTRAL, ls="--", lw=1)
    ax0.set_xlabel("1 − Specificity")
    ax0.set_ylabel("Sensitivity")
    ax0.set_title("ROC (external cohort)")
    ax0.legend(loc="lower right", frameon=False)
    ax0.set_xlim(-0.02, 1.02)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_aspect("equal", adjustable="box")

    # PR
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax1.plot(rec, prec, color=C_VAL, lw=2, label=f"AP = {ap:.3f}")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision–recall (external)")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Calibration
    prob_true, prob_pred = calibration_curve(y, s, n_bins=10, strategy="uniform")
    ax2.plot([0, 1], [0, 1], color=C_NEUTRAL, ls="--", lw=1, label="Ideal")
    ax2.plot(prob_pred, prob_true, marker="o", color=C_EXT, lw=1.5, label="Model")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Fraction of positives")
    ax2.set_title("Calibration (external)")
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle("External validation: discrimination and calibration", fontsize=10)
    save_both(fig, out / "pub_fig02_external_roc_pr_calibration")


def fig_confusion_enhanced(pred_csv: Path, out: Path) -> None:
    df = pd.read_csv(pred_csv, encoding="utf-8")
    y = df["label"].astype(int).values
    yhat = df["pred"].astype(int).values
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", color="black", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (external)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=90)
    fig.tight_layout()
    save_both(fig, out / "pub_fig03_external_confusion_matrix")


def fig_internal_external_bars(internal_json: Path, external_json: Path, out: Path) -> None:
    with open(internal_json, encoding="utf-8") as f:
        intd = json.load(f)
    with open(external_json, encoding="utf-8") as f:
        extd = json.load(f)

    val = intd["val_overall"]
    ext = extd["overall"]
    keys = [
        ("auc", "AUROC"),
        ("pr_auc", "PR-AUC"),
        ("balanced_acc", "Balanced accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    x = np.arange(len(keys))
    w = 0.36
    v_vals = [val[k] for k, _ in keys]
    e_vals = [ext[k] for k, _ in keys]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w / 2, v_vals, width=w, label="Internal validation", color=C_VAL, edgecolor="white", lw=0.5)
    ax.bar(x + w / 2, e_vals, width=w, label="External test", color=C_EXT, edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys], rotation=18, ha="right")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Overall performance: internal validation vs external test cohort")
    fig.tight_layout()
    save_both(fig, out / "pub_fig04_internal_vs_external_metrics")


def fig_external_per_site(per_center: Path, out: Path) -> None:
    raw = pd.read_csv(per_center, encoding="utf-8")
    d = raw.copy()
    d["auc_plot"] = pd.to_numeric(d["auc"], errors="coerce")
    d["n"] = d["n"].astype(int)
    d = d.sort_values("auc_plot", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.35 * len(d) + 1.2)))
    colors = [C_EXT if n >= 20 else C_NEUTRAL for n in d["n"]]
    ax.barh(y, d["auc_plot"].fillna(0), color=colors, height=0.65, edgecolor="white", lw=0.5)
    for i, row in enumerate(d.itertuples()):
        a = row.auc_plot
        label = f"{a:.3f}" if pd.notna(a) else "—"
        ax.text(min(0.98, (a if pd.notna(a) else 0) + 0.02), i, f"n={row.n}  {label}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["center_id"].astype(str), fontsize=8)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0, 1.05)
    ax.set_title("External test: AUROC by site (grey: n < 20)")
    leg = [
        Patch(facecolor=C_EXT, edgecolor="white", label="n ≥ 20"),
        Patch(facecolor=C_NEUTRAL, edgecolor="white", label="n < 20"),
    ]
    ax.legend(handles=leg, loc="lower right", frameon=False)
    fig.tight_layout()
    save_both(fig, out / "pub_fig05_external_auroc_by_site")


def fig_internal_val_per_site(per_center: Path, out: Path) -> None:
    raw = pd.read_csv(per_center, encoding="utf-8")
    d = raw.copy()
    d["auc_plot"] = pd.to_numeric(d["auc"], errors="coerce")
    d = d.sort_values("auc_plot", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.35 * len(d) + 1.2)))
    ax.barh(y, d["auc_plot"].fillna(0), color=C_VAL, height=0.65, edgecolor="white", lw=0.5)
    for i, row in enumerate(d.itertuples()):
        a = row.auc_plot
        label = f"{a:.3f}" if pd.notna(a) else "—"
        ax.text(min(0.98, (a if pd.notna(a) else 0) + 0.02), i, f"n={int(row.n)}  {label}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["center_id"].astype(str), fontsize=8)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0, 1.05)
    ax.set_title("Internal validation: AUROC by development centre")
    fig.tight_layout()
    save_both(fig, out / "pub_fig06_internal_val_auroc_by_site")


def fig_prob_by_label(pred_csv: Path, out: Path) -> None:
    df = pd.read_csv(pred_csv, encoding="utf-8")
    df = df.copy()
    df["label_str"] = df["label"].map({0: "Negative", 1: "Positive"})
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    parts = ax.violinplot(
        [df.loc[df["label"] == 0, "prob_pos"], df.loc[df["label"] == 1, "prob_pos"]],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
        widths=0.65,
    )
    for b in parts["bodies"]:
        b.set_facecolor(C_NEG)
        b.set_alpha(0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True negative", "True positive"])
    ax.set_ylabel("Predicted probability (positive class)")
    ax.set_title("External test: score distribution by outcome")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save_both(fig, out / "pub_fig07_external_score_violin")


def fig_metrics_heatmap(internal_json: Path, external_json: Path, out: Path) -> None:
    with open(internal_json, encoding="utf-8") as f:
        intd = json.load(f)
    with open(external_json, encoding="utf-8") as f:
        extd = json.load(f)
    rows = ["Internal train", "Internal val", "External test"]
    cols = ["AUROC", "PR-AUC", "Bal. acc.", "Sensitivity", "Specificity"]
    keys = [("auc",), ("pr_auc",), ("balanced_acc",), ("sensitivity",), ("specificity",)]
    mat = np.array(
        [
            [intd["train_overall"][k[0]] for k in keys],
            [intd["val_overall"][k[0]] for k in keys],
            [extd["overall"][k[0]] for k in keys],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticklabels(rows)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white" if mat[i, j] < 0.55 else "black", fontsize=8)
    ax.set_title("Key metrics overview (all quantities in [0, 1])")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Value")
    fig.tight_layout()
    save_both(fig, out / "pub_fig08_metrics_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-style figures for OCT triage (loc5out).")
    parser.add_argument(
        "--oct_root",
        type=str,
        default=None,
        help="experiments/OCT_traige root (default: parent of scripts/)",
    )
    parser.add_argument("--out_dir", type=str, default="logs/figures_publication")
    args = parser.parse_args()

    root = Path(args.oct_root) if args.oct_root else Path(__file__).resolve().parents[1]
    logs = root / "logs"
    out = root / args.out_dir
    apply_journal_style()

    mcsv = latest_metrics_csv(logs)
    pred = logs / "external_predictions.csv"
    int_json = logs / "internal_overall_metrics_loc5out.json"
    ext_json = logs / "external_metrics_loc5out.json"
    ext_pc = logs / "external_per_center_metrics_loc5out.csv"
    int_val_pc = logs / "internal_val_per_center_metrics_loc5out.csv"

    print(f"[info] metrics_history: {mcsv.name}")
    print(f"[info] output directory: {out}")

    fig_training_multipanel(mcsv, out)
    print("[saved] pub_fig01_training_curves")

    if pred.exists():
        fig_external_roc_pr_calib(pred, out)
        print("[saved] pub_fig02_external_roc_pr_calibration")
        fig_confusion_enhanced(pred, out)
        print("[saved] pub_fig03_external_confusion_matrix")
        fig_prob_by_label(pred, out)
        print("[saved] pub_fig07_external_score_violin")
    else:
        print("[skip] external_predictions.csv missing")

    if int_json.exists() and ext_json.exists():
        fig_internal_external_bars(int_json, ext_json, out)
        print("[saved] pub_fig04_internal_vs_external_metrics")
        fig_metrics_heatmap(int_json, ext_json, out)
        print("[saved] pub_fig08_metrics_heatmap")
    else:
        print("[skip] internal/external json missing")

    if ext_pc.exists():
        fig_external_per_site(ext_pc, out)
        print("[saved] pub_fig05_external_auroc_by_site")
    if int_val_pc.exists():
        fig_internal_val_per_site(int_val_pc, out)
        print("[saved] pub_fig06_internal_val_auroc_by_site")

    print("[done] Publication figures written to", out)


if __name__ == "__main__":
    main()
