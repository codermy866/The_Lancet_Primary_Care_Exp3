from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# 接近期刊印刷风格（无需额外字体文件）
rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Nimbus Roman"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def plot_training_history(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(epochs, df["train_loss"], label="Train", color="#1f4e79")
    axes[0].plot(epochs, df["val_loss"], label="Validation", color="#c55a11")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, df["train_auc"], label="Train", color="#1f4e79")
    axes[1].plot(epochs, df["val_auc"], label="Validation", color="#c55a11")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Area under ROC curve")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0.45, 1.02)

    axes[2].plot(epochs, df["train_f1"], label="Train", color="#1f4e79")
    axes[2].plot(epochs, df["val_f1"], label="Validation", color="#c55a11")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 score")
    axes[2].set_title("F1 (default threshold)")
    axes[2].legend(frameon=False)
    axes[2].set_ylim(-0.05, 1.05)

    fig.suptitle("Internal train/validation (LOC-5out, 5 development centres)", y=1.02, fontsize=12)
    fig.savefig(out_dir / "fig_loc5out_training_curves.png")
    plt.close(fig)


def plot_external_roc_pr_cm(pred_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(pred_csv, encoding="utf-8")
    y = df["label"].astype(int).values
    s = df["prob_pos"].astype(float).values
    yhat = df["pred"].astype(int).values

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    fpr, tpr, _ = roc_curve(y, s)
    axes[0].plot(fpr, tpr, color="#1f4e79", lw=2, label=f"AUROC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    axes[0].set_xlabel("1 − Specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("ROC (external test)")
    axes[0].legend(loc="lower right", frameon=False)

    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    axes[1].plot(rec, prec, color="#c55a11", lw=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall (external test)")
    axes[1].legend(loc="upper right", frameon=False)

    cm = confusion_matrix(y, yhat, labels=[0, 1])
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Pred 0", "Pred 1"])
    axes[2].set_yticklabels(["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        axes[2].text(j, i, int(v), ha="center", va="center", color="black", fontsize=12)
    axes[2].set_title("Confusion matrix (external)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("External validation (5 held-out centres)", y=1.02, fontsize=12)
    fig.savefig(out_dir / "fig_loc5out_external_roc_pr_cm.png")
    plt.close(fig)


def plot_external_per_center(per_center_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(per_center_csv, encoding="utf-8").sort_values("n", ascending=False)
    x = np.arange(len(df))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, df["auc"], width=w, label="AUROC", color="#1f4e79")
    ax.bar(x + w / 2, df["f1"], width=w, label="F1", color="#c55a11")
    ax.set_xticks(x)
    ax.set_xticklabels(df["center_id"], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("External performance by centre (default threshold)")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "fig_loc5out_external_per_centre.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default="logs/metrics_history_20260402_223023.csv")
    parser.add_argument("--pred_csv", type=str, default="logs/external_predictions.csv")
    parser.add_argument("--per_center_csv", type=str, default="logs/external_per_center_metrics_loc5out.csv")
    parser.add_argument("--out_dir", type=str, default="logs/figures_loc5out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mcsv = root / args.metrics_csv
    if mcsv.exists():
        plot_training_history(mcsv, out_dir)
        print(f"[saved] {out_dir / 'fig_loc5out_training_curves.png'}")
    else:
        print(f"[skip] missing {mcsv}")

    pcsv = root / args.pred_csv
    if pcsv.exists():
        plot_external_roc_pr_cm(pcsv, out_dir)
        print(f"[saved] {out_dir / 'fig_loc5out_external_roc_pr_cm.png'}")
    else:
        print(f"[skip] missing {pcsv} (run eval_external_oct_traige first)")

    ccsv = root / args.per_center_csv
    if ccsv.exists():
        plot_external_per_center(ccsv, out_dir)
        print(f"[saved] {out_dir / 'fig_loc5out_external_per_centre.png'}")
    else:
        print(f"[skip] missing {ccsv}")


if __name__ == "__main__":
    main()
