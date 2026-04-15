"""
高级小提琴图 + 云雨图（Raincloud）可视化。

依赖：
  pip install ptitprince seaborn

在 experiments/OCT_traige 下运行：
  python scripts/violin_raincloud_loc5out.py
  python scripts/violin_raincloud_loc5out.py --out_dir logs/figures_publication

输出（PNG 300dpi + PDF）：
  - pub_raincloud_outcome.*                    : 外部 真阴/真阳 云雨图（ptitprince）
  - pub_violin_advanced_outcome.*             : 外部 增强小提琴
  - pub_raincloud_manual_by_site.*            : 外部各中心横向云雨
  - pub_split_violin_site_label.*             : 外部各中心 split violin
  - pub_raincloud_manual_internal_5sites.*    : 内部验证 5 家医院（同一张横向云雨图）
  - pub_split_violin_internal_5sites.*        : 内部验证 5 家 split violin
  - pub_raincloud_internal_external_twopanel.* : 内部 5 中心 + 外部中心 上下双面板（可选对比）
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

# 复用顶刊脚本中的字体与保存
_SCRIPT_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_pub_fig",
    _SCRIPT_DIR / "publication_figures_loc5out.py",
)
_pub = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_pub)
apply_journal_style = _pub.apply_journal_style
save_both = _pub.save_both

C_NEG = "#0072B2"
C_POS = "#E69F00"

# 顶刊 Dusty rose + sage（与 exp3_manuscript_exp 一致）
JOURNAL_PALETTE = ("#C18C8A", "#bea597", "#bfa195", "#dde2d3", "#ccd4bd")
# pub_raincloud_outcome：真阴 / 真阳
RC_OUTCOME_NEG = JOURNAL_PALETTE[3]
RC_OUTCOME_POS = JOURNAL_PALETTE[0]


def _load_pred(logs: Path) -> pd.DataFrame:
    p = logs / "external_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, encoding="utf-8")
    df["outcome"] = df["label"].map({0: "True negative", 1: "True positive"})
    df["site"] = df["center_id_external"].astype(str)
    return df


def _load_internal_val(logs: Path) -> pd.DataFrame:
    """内部验证集逐样本预测（5 家开发医院，不含 train 里可能出现的未知中心）。"""
    p = logs / "internal_val_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, usecols=["label", "center_id", "prob_pos"])
    df["outcome"] = df["label"].map({0: "True negative", 1: "True positive"})
    df["site"] = df["center_id"].astype(str)
    return df


def fig_raincloud_outcome(df: Path, out: Path) -> None:
    import ptitprince as pt

    d = _load_pred(df.parent)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pt.RainCloud(
            x="outcome",
            y="prob_pos",
            data=d,
            order=["True negative", "True positive"],
            orient="v",
            width_viol=0.56,
            width_box=0.13,
            move=0.17,
            bw=0.22,
            linewidth=0.65,
            palette=[RC_OUTCOME_NEG, RC_OUTCOME_POS],
            ax=ax,
            # 雨点：略透明 + 白描边，层次更清晰（点大小用 RainCloud 的 point_size）
            point_size=3.2,
            rain_alpha=0.52,
            rain_edgecolor="white",
            rain_linewidth=0.35,
            # 箱线：与调色盘协调的深灰褐
            box_linewidth=1.05,
            box_medianprops={"color": "#4a423d", "linewidth": 1.35},
            box_whiskerprops={"linewidth": 0.95, "color": "#6d6560"},
            box_capprops={"linewidth": 0.95, "color": "#6d6560"},
        )
    ax.set_xlabel("")
    ax.set_ylabel("Predicted probability (positive class)")
    ax.set_title("External test: raincloud by outcome")
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.45)
    fig.tight_layout()
    save_both(fig, out / "pub_raincloud_outcome")


def fig_violin_advanced_outcome(df: Path, out: Path) -> None:
    d = _load_pred(df.parent)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    sns.violinplot(
        data=d,
        x="outcome",
        y="prob_pos",
        hue="outcome",
        order=["True negative", "True positive"],
        hue_order=["True negative", "True positive"],
        palette=[C_NEG, C_POS],
        inner="quartile",
        cut=0,
        linewidth=0.8,
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        data=d.sample(min(4000, len(d)), random_state=0),
        x="outcome",
        y="prob_pos",
        order=["True negative", "True positive"],
        color="0.25",
        alpha=0.22,
        size=1.8,
        jitter=0.22,
        ax=ax,
    )
    sns.boxplot(
        data=d,
        x="outcome",
        y="prob_pos",
        order=["True negative", "True positive"],
        width=0.12,
        showcaps=True,
        boxprops={"zorder": 3, "facecolor": "none"},
        whiskerprops={"linewidth": 1.0},
        medianprops={"color": "0.15", "linewidth": 1.4},
        ax=ax,
        zorder=4,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Predicted probability (positive class)")
    ax.set_title("External test: enhanced violin (quartiles + box + strip)")
    fig.tight_layout()
    save_both(fig, out / "pub_violin_advanced_outcome")


def _half_violin_site(
    ax,
    data: np.ndarray,
    y_center: float,
    y_half_width: float,
    color: str,
    alpha: float = 0.55,
) -> None:
    """x=概率, y=中心索引：在 y_center 上方叠一层核密度「云」（向 +y 展开）。"""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return
    kde = gaussian_kde(data)
    x_grid = np.linspace(-0.02, 1.02, 256)
    dens = kde(x_grid)
    dens = dens / (dens.max() + 1e-9) * y_half_width
    ax.fill_between(x_grid, y_center, y_center + dens, color=color, alpha=alpha, lw=0, zorder=1)


def _draw_raincloud_manual_horizontal(
    ax: plt.Axes,
    d: pd.DataFrame,
    site_col: str,
    prob_col: str,
    order: list,
    y_axis_label: str,
    panel_title: str,
) -> None:
    """在 ax 上绘制「横向云雨」：一行一个中心。"""
    n = len(order)
    y_half = 0.32
    colors = sns.color_palette("Set2", n_colors=max(n, 3))[:n]

    for i, site in enumerate(order):
        sub = d.loc[d[site_col] == site, prob_col].values.astype(float)
        y0 = float(i)
        c = colors[i % len(colors)]
        _half_violin_site(ax, sub, y0, y_half, c, alpha=0.5)
        y_box = y0 + y_half + 0.12
        ax.boxplot(
            [sub],
            positions=[y_box],
            vert=False,
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.1", "linewidth": 1.3},
            boxprops={"facecolor": "white", "edgecolor": "0.3", "linewidth": 0.9},
            whiskerprops={"linewidth": 0.9, "color": "0.35"},
            capprops={"linewidth": 0.9, "color": "0.35"},
        )
        rng = np.random.default_rng(hash(str(site)) % (2**32))
        jit_y = rng.uniform(-0.06, 0.06, size=sub.size)
        ax.scatter(
            sub,
            y_box + 0.16 + jit_y,
            s=6,
            alpha=0.3,
            c=[c],
            edgecolors="none",
            zorder=2,
        )
        ax.text(
            1.02,
            y0 + y_half * 0.5,
            f"n={len(sub)}",
            va="center",
            fontsize=8,
            color="0.35",
        )

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(order)
    ax.set_xlabel("Predicted probability (positive class)")
    ax.set_ylabel(y_axis_label)
    ax.set_xlim(-0.05, 1.18)
    ax.set_title(panel_title)


def fig_raincloud_manual_by_site(df: Path, out: Path) -> None:
    d = _load_pred(df.parent)
    counts = d.groupby("site").size().sort_values(ascending=True)
    order = counts.index.tolist()
    n = len(order)
    fig_h = max(3.6, 0.72 * n + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    _draw_raincloud_manual_horizontal(
        ax,
        d,
        "site",
        "prob_pos",
        order,
        "External site",
        "External test: raincloud-style plot by site (half-violin + box + jitter)",
    )
    fig.tight_layout()
    save_both(fig, out / "pub_raincloud_manual_by_site")


def fig_raincloud_manual_internal_5sites(logs: Path, out: Path) -> None:
    d = _load_internal_val(logs)
    counts = d.groupby("site").size().sort_values(ascending=True)
    order = counts.index.tolist()
    n = len(order)
    fig_h = max(4.0, 0.72 * n + 1.6)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    _draw_raincloud_manual_horizontal(
        ax,
        d,
        "site",
        "prob_pos",
        order,
        "Hospital (internal validation)",
        "Internal validation: 5 development centres on one panel",
    )
    fig.tight_layout()
    save_both(fig, out / "pub_raincloud_manual_internal_5sites")


def fig_raincloud_internal_external_twopanel(logs: Path, out: Path) -> None:
    """内部 5 中心与外部中心上下排列，便于同刊并排对比。"""
    d_int = _load_internal_val(logs)
    d_ext = _load_pred(logs)
    order_int = d_int.groupby("site").size().sort_values(ascending=True).index.tolist()
    order_ext = d_ext.groupby("site").size().sort_values(ascending=True).index.tolist()
    n_rows = len(order_int) + len(order_ext)
    fig_h = max(6.5, 0.55 * n_rows + 2.2)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, fig_h), sharex=True)
    _draw_raincloud_manual_horizontal(
        axes[0],
        d_int,
        "site",
        "prob_pos",
        order_int,
        "Hospital",
        "(A) Internal validation — five development centres",
    )
    _draw_raincloud_manual_horizontal(
        axes[1],
        d_ext,
        "site",
        "prob_pos",
        order_ext,
        "External site",
        "(B) External test — held-out centres",
    )
    fig.suptitle("Predicted probability of positive class by centre", fontsize=10, y=1.01)
    fig.tight_layout()
    save_both(fig, out / "pub_raincloud_internal_external_twopanel")


def fig_split_violin_site_label(df: Path, out: Path) -> None:
    d = _load_pred(df.parent)
    counts = d.groupby("site").size().sort_values(ascending=False)
    order = counts.index.tolist()
    fig_w = max(7.0, 1.1 * len(order) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    sns.violinplot(
        data=d,
        x="site",
        y="prob_pos",
        hue="outcome",
        order=order,
        hue_order=["True negative", "True positive"],
        split=True,
        inner="quartile",
        palette=[C_NEG, C_POS],
        cut=0,
        linewidth=0.7,
        ax=ax,
    )
    ax.legend(title="", loc="upper right", frameon=False)
    ax.set_xlabel("External site")
    ax.set_ylabel("Predicted probability")
    ax.set_title("External test: split violin by outcome within each site")
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    save_both(fig, out / "pub_split_violin_site_label")


def fig_split_violin_internal_5sites(logs: Path, out: Path) -> None:
    d = _load_internal_val(logs)
    counts = d.groupby("site").size().sort_values(ascending=False)
    order = counts.index.tolist()
    fig_w = max(8.0, 1.15 * len(order) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    sns.violinplot(
        data=d,
        x="site",
        y="prob_pos",
        hue="outcome",
        order=order,
        hue_order=["True negative", "True positive"],
        split=True,
        inner="quartile",
        palette=[C_NEG, C_POS],
        cut=0,
        linewidth=0.7,
        ax=ax,
    )
    ax.legend(title="", loc="upper right", frameon=False)
    ax.set_xlabel("Hospital (internal validation)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Internal validation: split violin by outcome — five centres on one figure")
    plt.xticks(rotation=22, ha="right")
    fig.tight_layout()
    save_both(fig, out / "pub_split_violin_internal_5sites")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oct_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="logs/figures_publication")
    args = parser.parse_args()

    root = Path(args.oct_root) if args.oct_root else Path(__file__).resolve().parents[1]
    logs = root / "logs"
    out = root / args.out_dir
    pred = logs / "external_predictions.csv"
    internal_val = logs / "internal_val_predictions.csv"

    apply_journal_style()

    try:
        import ptitprince  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "需要安装: pip install ptitprince seaborn\n" + str(e)
        ) from e

    print(f"[info] output -> {out}")
    fig_raincloud_outcome(pred, out)
    print("[saved] pub_raincloud_outcome")
    fig_violin_advanced_outcome(pred, out)
    print("[saved] pub_violin_advanced_outcome")
    fig_raincloud_manual_by_site(pred, out)
    print("[saved] pub_raincloud_manual_by_site")
    fig_split_violin_site_label(pred, out)
    print("[saved] pub_split_violin_site_label")

    if internal_val.exists():
        fig_raincloud_manual_internal_5sites(logs, out)
        print("[saved] pub_raincloud_manual_internal_5sites")
        fig_split_violin_internal_5sites(logs, out)
        print("[saved] pub_split_violin_internal_5sites")
        fig_raincloud_internal_external_twopanel(logs, out)
        print("[saved] pub_raincloud_internal_external_twopanel")
    else:
        print("[skip] internal_val_predictions.csv missing (run eval_internal_oct_traige.py)")

    print("[done]")


if __name__ == "__main__":
    main()
