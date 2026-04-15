"""
Exp 3.3 — Multicenter generalizability & subgroup utility（稿件级输出）

默认（内部 vs 外部整体）：
  - Table：内部验证集 / 外部测试集 各一行：n、AUROC(95%CI)、固定阈值下 Sens/Spec/PPV/NPV、Youden J、LR+/LR-
  - Fig：点估计对比（AUROC+CI + Sens/Spec/Youden 柱图）
  - Fig（分布类）：预测概率 split 小提琴、bootstrap AUROC 小提琴、云雨图、KDE 密度叠加
  - Fig：joint 布局 — 预测概率 vs 结局（抖动）+ 顶部边际 KDE（内外分色）
  - 若 --clinical_csv 可与内外 oct_id 同时合并：另存「队列 × 亚组」split violin（HPV / 年龄 / TCT）

可选：
  - --per_centre_tables：另存逐中心 CSV/LaTeX（不做逐医院图）
  - --clinical_csv：亚组（HPV/年龄/TCT 等）若可与 oct_id 合并则输出；否则写 merge 报告

用法（在 experiments/OCT_traige 下）:
  python scripts/exp3_manuscript_exp.py
  python scripts/exp3_manuscript_exp.py --per_centre_tables
  python scripts/exp3_manuscript_exp.py --clinical_csv /path/to/master_with_OCT.csv

说明：
  - 「OBR」在稿件中若另有定义请替换；此处用 Youden J（Sens+Spec−1）作为单阈值下的综合区分指标。
  - 6 家全外部或 LOHO 需重新划分数据并重训，本脚本不自动跑 LOHO，仅产出当前 loc5out 评估结果。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_pub", _SCRIPT_DIR / "publication_figures_loc5out.py")
_pub = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_pub)
apply_journal_style = _pub.apply_journal_style
save_both = _pub.save_both

_SPEC2 = importlib.util.spec_from_file_location("_st", _SCRIPT_DIR / "statistical_supplement_loc5out.py")
_st = importlib.util.module_from_spec(_SPEC2)
assert _SPEC2.loader is not None
_SPEC2.loader.exec_module(_st)
bootstrap_auc_ci = _st.bootstrap_auc_ci

# 顶刊 Dusty rose + sage 配色（用户指定）
JOURNAL_PALETTE = ("#C18C8A", "#bea597", "#bfa195", "#dde2d3", "#ccd4bd")
COLOR_COHORT: tuple[str, str] = (JOURNAL_PALETTE[0], JOURNAL_PALETTE[1])  # Internal, External
COLOR_OUTCOME_SPLIT = {"Negative": JOURNAL_PALETTE[3], "Positive": JOURNAL_PALETTE[0]}
RAIN_STRIP = "#5a524c"


def metrics_at_threshold(y: np.ndarray, prob: np.ndarray, thr: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    lr_pos = sens / (1 - spec) if spec < 1 and np.isfinite(spec) and spec == spec else float("nan")
    lr_neg = (1 - sens) / spec if spec > 0 and np.isfinite(sens) else float("nan")
    youden = sens + spec - 1 if np.isfinite(sens) and np.isfinite(spec) else float("nan")
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "youden_j": float(youden),
        "lr_positive": float(lr_pos),
        "lr_negative": float(lr_neg),
    }


def choose_threshold_from_internal(
    y: np.ndarray,
    prob: np.ndarray,
    strategy: str,
    fallback: float,
) -> float:
    """在内部验证集上选择部署阈值，避免使用外部标签调参。"""
    if strategy == "fixed":
        return float(fallback)
    y = np.asarray(y, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return float(fallback)
    # 网格搜索稳健阈值：用于提升固定工作点指标（AUC 不受阈值影响）
    grid = np.linspace(0.2, 0.8, 601)
    best_t = float(fallback)
    best_score = -1e9
    for t in grid:
        m = metrics_at_threshold(y, p, float(t))
        score = float(m["youden_j"])
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _hpv_16_18(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    if t in ("", "-", "nan"):
        return None
    if re.search(r"(?<![0-9])16(?![0-9])", t) or re.search(r"(?<![0-9])18(?![0-9])", t):
        return "HPV16/18"
    return "non_16_18"


def _tct_binary(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().upper()
    if t in ("", "-", "NAN"):
        return None
    if t == "NILM":
        return "NILM"
    return "Abnormal_cytology"


def summarize_cohort_overall(
    pred: pd.DataFrame,
    thr: float,
    n_boot: int,
    seed: int,
    cohort: str,
) -> dict[str, Any]:
    y = pred["label"].astype(int).values
    pr = pred["prob_pos"].astype(float).values
    boot = bootstrap_auc_ci(y, pr, n_boot=n_boot, seed=seed)
    m = metrics_at_threshold(y, pr, thr)
    auc_val = float("nan")
    if len(np.unique(y)) >= 2:
        try:
            auc_val = float(roc_auc_score(y, pr))
        except ValueError:
            auc_val = float("nan")
    return {
        "cohort": cohort,
        "n": len(pred),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": thr,
        "auroc": auc_val,
        "auc_ci_low": boot["auc_ci_low"],
        "auc_ci_high": boot["auc_ci_high"],
        **m,
    }


def plot_internal_vs_external(overall: pd.DataFrame, out_base: Path, title: str) -> None:
    """overall: 两行，含 cohort / auroc / auc_ci_* / sensitivity / specificity / youden_j。"""
    order = ["Internal validation", "External test"]
    cohort_to_label = {"internal_validation": order[0], "external_test": order[1]}
    df = overall.copy()
    df["display"] = df["cohort"].astype(str).map(lambda c: cohort_to_label.get(c, c))
    labels = [x for x in order if x in set(df["display"].values)]
    if not labels:
        return
    df = df.set_index("display").reindex(labels).dropna(how="all")
    if df.empty:
        return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = np.arange(len(df))
    x_auroc = df["auroc"].values.astype(float)
    err_lo = x_auroc - df["auc_ci_low"].values.astype(float)
    err_hi = df["auc_ci_high"].values.astype(float) - x_auroc
    bar_cols = [COLOR_COHORT[i % len(COLOR_COHORT)] for i in range(len(y))]
    ax0.barh(y, x_auroc, height=0.55, color=bar_cols, alpha=0.92, edgecolor="white", linewidth=0.6)
    ax0.errorbar(x_auroc, y, xerr=[err_lo, err_hi], fmt="none", ecolor="#6d6560", capsize=3, lw=0.9)
    ax0.set_yticks(y)
    ax0.set_yticklabels(list(df.index))
    ax0.set_xlabel("AUROC")
    ax0.set_xlim(0, 1.02)
    ax0.set_title("Discrimination (bootstrap 95% CI)")
    ax0.axvline(0.5, color=JOURNAL_PALETTE[4], ls=":", lw=0.85)

    metrics = ["sensitivity", "specificity", "youden_j"]
    labels_m = ["Sensitivity", "Specificity", "Youden J"]
    x = np.arange(len(metrics))
    w = 0.36
    all_vals: list[float] = []
    for i, row in enumerate(df.itertuples()):
        vals = [float(getattr(row, m)) for m in metrics]
        all_vals.extend(vals)
        off = -w / 2 if i == 0 else w / 2
        ax1.bar(
            x + off,
            vals,
            width=w,
            label=str(row.Index),
            color=COLOR_COHORT[i % 2],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.6,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_m, rotation=15, ha="right")
    lo = min(0.0, float(np.nanmin(all_vals)) - 0.05)
    hi = max(1.05, float(np.nanmax(all_vals)) + 0.05)
    ax1.set_ylim(lo, hi)
    ax1.set_ylabel("Value")
    thr0 = float(df["threshold"].iloc[0])
    ax1.set_title(f"Operating point (threshold = {thr0:g})")
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    save_both(fig, out_base)


def bootstrap_auc_values(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """与 statistical_supplement 一致的 bootstrap 重抽样 AUROC 序列（用于分布图）。"""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    store: list[float] = []
    if n < 2 or len(np.unique(y_true)) < 2:
        return np.array([], dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        store.append(float(roc_auc_score(yt, prob[idx])))
    return np.asarray(store, dtype=float)


def build_pred_long_df(intv: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in intv.iterrows():
        rows.append(
            {
                "cohort": "Internal validation",
                "label": int(r["label"]),
                "prob_pos": float(r["prob_pos"]),
            }
        )
    for _, r in ext.iterrows():
        rows.append(
            {
                "cohort": "External test",
                "label": int(r["label"]),
                "prob_pos": float(r["prob_pos"]),
            }
        )
    df = pd.DataFrame(rows)
    df["outcome"] = df["label"].map({0: "Negative", 1: "Positive"})
    cat_order = ["Internal validation", "External test"]
    df["cohort"] = pd.Categorical(df["cohort"], categories=cat_order, ordered=True)
    return df


def _raincloud_one(
    ax: plt.Axes,
    y: np.ndarray,
    x_pos: float,
    color: str,
    cloud_width: float,
    rng: np.random.Generator,
) -> None:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        ax.scatter([x_pos], [float(np.nanmean(y)) if len(y) else 0.5], c=color, s=40, zorder=5)
        return
    kde = gaussian_kde(y)
    yy = np.linspace(float(y.min()), float(y.max()), 80)
    dd = kde(yy)
    dd = dd / (dd.max() + 1e-12) * cloud_width
    ax.fill_betweenx(yy, x_pos, x_pos + dd, color=color, alpha=0.42, linewidth=0, zorder=1)
    ax.boxplot(
        [y],
        positions=[x_pos],
        widths=0.09,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": color, "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 0.9},
        whiskerprops={"color": color, "linewidth": 0.8},
        capprops={"color": color, "linewidth": 0.8},
    )
    jitter = rng.uniform(-0.07, 0.07, size=len(y))
    ax.scatter(
        np.full(len(y), x_pos - 0.12) + jitter,
        y,
        alpha=0.34,
        s=9,
        color=RAIN_STRIP,
        lw=0,
        zorder=3,
    )


def plot_exp3_distribution_figs(
    intv: pd.DataFrame,
    ext: pd.DataFrame,
    figs: Path,
    n_boot: int,
    seed_int: int,
    seed_ext: int,
    title_prefix: str = "OCT+AI",
) -> None:
    """小提琴、bootstrap 小提琴、云雨图、KDE 叠加。"""
    df_long = build_pred_long_df(intv, ext)
    cohort_order = ["Internal validation", "External test"]
    pal_out = COLOR_OUTCOME_SPLIT

    # 1) Split violin：预测概率，按结局分层（若某队列缺一类则退化为非 split）
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.8))
    split_ok = True
    for coh in cohort_order:
        sub = df_long[df_long["cohort"] == coh]
        if sub["label"].nunique() < 2:
            split_ok = False
            break
    try:
        if split_ok:
            sns.violinplot(
                data=df_long,
                x="cohort",
                y="prob_pos",
                hue="outcome",
                order=cohort_order,
                hue_order=["Negative", "Positive"],
                split=True,
                inner="quartile",
                palette=pal_out,
                linewidth=0.9,
                ax=ax1,
            )
        else:
            sns.violinplot(
                data=df_long,
                x="cohort",
                y="prob_pos",
                hue="outcome",
                order=cohort_order,
                inner="quartile",
                palette=pal_out,
                linewidth=0.9,
                ax=ax1,
            )
    except (ValueError, RuntimeError):
        sns.violinplot(
            data=df_long,
            x="cohort",
            y="prob_pos",
            order=cohort_order,
            color=JOURNAL_PALETTE[2],
            inner="quartile",
            linewidth=0.9,
            ax=ax1,
        )
    ax1.set_xlabel("")
    ax1.set_ylabel("Predicted probability (positive class)")
    ax1.set_title(f"{title_prefix}: score distribution by outcome")
    leg = ax1.get_legend()
    if leg is not None:
        leg.set_title("True label")
    ax1.set_ylim(-0.02, 1.02)
    fig1.tight_layout()
    save_both(fig1, figs / "manuscript_fig_exp3_violin_prob_split_outcome")

    # 2) Bootstrap AUROC 分布（小提琴）
    y_i = intv["label"].astype(int).values
    p_i = intv["prob_pos"].astype(float).values
    y_e = ext["label"].astype(int).values
    p_e = ext["prob_pos"].astype(float).values
    boot_i = bootstrap_auc_values(y_i, p_i, n_boot=n_boot, seed=seed_int)
    boot_e = bootstrap_auc_values(y_e, p_e, n_boot=n_boot, seed=seed_ext)
    if len(boot_i) and len(boot_e):
        df_boot = pd.DataFrame(
            {
                "cohort": np.concatenate(
                    [np.full(len(boot_i), cohort_order[0]), np.full(len(boot_e), cohort_order[1])]
                ),
                "auroc_boot": np.concatenate([boot_i, boot_e]),
            }
        )
        df_boot["cohort"] = pd.Categorical(df_boot["cohort"], categories=cohort_order, ordered=True)
        fig2, ax2 = plt.subplots(figsize=(4.8, 3.8))
        sns.violinplot(
            data=df_boot,
            x="cohort",
            y="auroc_boot",
            hue="cohort",
            order=cohort_order,
            hue_order=cohort_order,
            palette=list(COLOR_COHORT),
            inner="box",
            linewidth=0.9,
            legend=False,
            ax=ax2,
        )
        ax2.axhline(0.5, color="#c9c4bf", ls=":", lw=0.85)
        ax2.set_xlabel("")
        ax2.set_ylabel("Bootstrap AUROC")
        ax2.set_title(f"{title_prefix}: AUROC uncertainty (bootstrap n={n_boot})")
        ax2.set_ylim(0.0, 1.02)
        fig2.tight_layout()
        save_both(fig2, figs / "manuscript_fig_exp3_violin_bootstrap_auroc")

    # 3) 云雨图（strip + box + 半小提琴）：整体预测概率
    fig3, ax3 = plt.subplots(figsize=(5.0, 3.8))
    rng = np.random.default_rng(seed_int + 1000)
    colors_c = list(COLOR_COHORT)
    for i, coh in enumerate(cohort_order):
        sub = df_long[df_long["cohort"] == coh]["prob_pos"].values
        _raincloud_one(ax3, sub, float(i), colors_c[i], cloud_width=0.32, rng=rng)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(cohort_order)
    ax3.set_ylabel("Predicted probability (positive class)")
    ax3.set_title(f"{title_prefix}: raincloud (strip + box + density)")
    ax3.set_xlim(-0.45, 1.45)
    ax3.set_ylim(-0.02, 1.02)
    fig3.tight_layout()
    save_both(fig3, figs / "manuscript_fig_exp3_raincloud_prob")

    # 4) KDE：内外部预测概率边际分布
    fig4, ax4 = plt.subplots(figsize=(5.0, 3.6))
    for coh, color in zip(cohort_order, colors_c):
        sub = df_long[df_long["cohort"] == coh]["prob_pos"].values
        if len(sub) < 2:
            continue
        sns.kdeplot(
            x=sub,
            ax=ax4,
            color=color,
            fill=True,
            alpha=0.38,
            linewidth=1.35,
            label=coh,
        )
    ax4.set_xlabel("Predicted probability (positive class)")
    ax4.set_ylabel("Density")
    ax4.set_title(f"{title_prefix}: predicted probability density")
    ax4.legend(loc="upper right", frameon=False)
    ax4.set_xlim(-0.02, 1.02)
    fig4.tight_layout()
    save_both(fig4, figs / "manuscript_fig_exp3_kde_prob_density")


def plot_joint_prob_outcome_marginal(
    df_long: pd.DataFrame,
    out_base: Path,
    title_prefix: str,
    seed: int,
) -> None:
    """主图：预测概率 vs 真值标签（y 方向抖动）；顶部边际：各队列预测概率 KDE。"""
    rng = np.random.default_rng(seed)
    cohort_order = ["Internal validation", "External test"]
    colors_c = list(COLOR_COHORT)
    df = df_long.copy()
    df["y_j"] = df["label"].astype(float) + rng.uniform(-0.08, 0.08, size=len(df))
    fig = plt.figure(figsize=(5.7, 4.95), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 4.0], hspace=0.07)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    for coh, c in zip(cohort_order, colors_c):
        sub = df[df["cohort"] == coh]
        if len(sub) == 0:
            continue
        ax_main.scatter(
            sub["prob_pos"],
            sub["y_j"],
            c=c,
            alpha=0.38,
            s=11,
            label=coh,
            linewidths=0.35,
            edgecolors="white",
            rasterized=True,
        )
        xk = sub["prob_pos"].astype(float).values
        if len(xk) >= 2 and np.nanstd(xk) > 1e-8:
            sns.kdeplot(
                x=xk,
                ax=ax_top,
                color=c,
                fill=True,
                alpha=0.42,
                lw=1.25,
                clip=(0, 1),
                common_norm=False,
            )
    ax_top.set_xlim(-0.02, 1.02)
    ax_top.set_ylabel("Density")
    ax_top.set_xticklabels([])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_main.set_xlabel("Predicted probability (positive class)")
    ax_main.set_ylabel("True label (jittered)")
    ax_main.set_yticks([0, 1])
    ax_main.set_yticklabels(["Negative (0)", "Positive (1)"])
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(-0.38, 1.38)
    ax_main.legend(loc="upper left", frameon=False, fontsize=8)
    ax_main.set_title(f"{title_prefix}: probability vs outcome (marginal density on top)")
    save_both(fig, out_base)


def merge_internal_external_clinical(
    intv: pd.DataFrame,
    ext: pd.DataFrame,
    clinical: pd.DataFrame,
    oct_key_pred: str,
    oct_key_clin: str,
) -> pd.DataFrame | None:
    """将内部验证 + 外部测试与临床表按 OCT/oct_id 合并，并派生亚组列。"""
    clin = clinical.copy()
    clin = clin.rename(columns={oct_key_clin: "_oct_merge"})
    mi = intv.merge(clin, left_on=oct_key_pred, right_on="_oct_merge", how="inner")
    me = ext.merge(clin, left_on=oct_key_pred, right_on="_oct_merge", how="inner")
    mi = mi.drop(columns=["_oct_merge"], errors="ignore")
    me = me.drop(columns=["_oct_merge"], errors="ignore")
    mi["cohort"] = "Internal validation"
    me["cohort"] = "External test"
    if len(mi) == 0 and len(me) == 0:
        return None
    out = pd.concat([mi, me], ignore_index=True)
    if "AGE" in out.columns:
        med = float(out["AGE"].median())
        out["age_group"] = np.where(out["AGE"] >= med, f"age>={med:g}", f"age<{med:g}")
    if "HPV清洗" in out.columns:
        out["hpv_16_18"] = out["HPV清洗"].map(_hpv_16_18)
    if "TCT清洗" in out.columns:
        out["tct_group"] = out["TCT清洗"].map(_tct_binary)
    return out


def plot_cohort_x_subgroup_split_violins(
    merged_ie: pd.DataFrame,
    figs: Path,
    title_prefix: str,
    min_n: int = 12,
) -> list[str]:
    """内外 × 亚组：x=队列，hue=亚组；二元亚组用 split violin。"""
    cohort_order = ["Internal validation", "External test"]
    merged_ie = merged_ie.copy()
    merged_ie["outcome"] = merged_ie["label"].map({0: "Negative", 1: "Positive"})
    merged_ie["cohort"] = pd.Categorical(merged_ie["cohort"], categories=cohort_order, ordered=True)
    saved: list[str] = []
    spec: list[tuple[str, str, list[str] | None]] = [
        ("hpv_16_18", "HPV 16/18 vs other", ["HPV16/18", "non_16_18"]),
        ("age_group", "Age (median split)", None),
        ("tct_group", "TCT NILM vs abnormal", ["NILM", "Abnormal_cytology"]),
    ]
    for col, title_suffix, hue_order in spec:
        if col not in merged_ie.columns:
            continue
        subdf = merged_ie.dropna(subset=[col]).copy()
        if len(subdf) < min_n:
            continue
        if subdf["cohort"].nunique() < 2:
            continue
        vc = subdf[col].astype(str).value_counts()
        if (vc < 3).any():
            continue
        levels = sorted(subdf[col].astype(str).unique())
        use_split = len(levels) == 2
        fig, ax = plt.subplots(figsize=(5.9, 4.0))
        pal2 = [JOURNAL_PALETTE[0], JOURNAL_PALETTE[2]]
        try:
            if use_split:
                ho = hue_order if hue_order and set(levels).issubset(set(hue_order)) else levels
                sns.violinplot(
                    data=subdf,
                    x="cohort",
                    y="prob_pos",
                    hue=col,
                    order=cohort_order,
                    hue_order=ho,
                    split=True,
                    inner="quartile",
                    palette=pal2,
                    linewidth=0.9,
                    ax=ax,
                )
            else:
                sns.violinplot(
                    data=subdf,
                    x="cohort",
                    y="prob_pos",
                    hue=col,
                    order=cohort_order,
                    inner="quartile",
                    linewidth=0.9,
                    ax=ax,
                )
        except (ValueError, RuntimeError):
            sns.violinplot(
                data=subdf,
                x="cohort",
                y="prob_pos",
                order=cohort_order,
                color=JOURNAL_PALETTE[2],
                inner="quartile",
                ax=ax,
            )
        ax.set_xlabel("")
        ax.set_ylabel("Predicted probability (positive class)")
        ax.set_title(f"{title_prefix}: {title_suffix}")
        ax.set_ylim(-0.02, 1.02)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(col.replace("_", " "))
        fig.tight_layout()
        safe = col.replace("/", "_").replace(" ", "_")
        stem = figs / f"manuscript_fig_exp3_violin_cohort_x_{safe}"
        save_both(fig, stem)
        saved.append(str(stem.with_suffix(".png")))
    return saved


def build_merged_ie_from_clinical(
    intv: pd.DataFrame,
    ext: pd.DataFrame,
    clin_path: Path,
    oct_key_clin: str,
) -> tuple[pd.DataFrame | None, str]:
    if not clin_path.is_file():
        return None, "clinical file missing"
    clin = pd.read_csv(clin_path, low_memory=False)
    oct_key_pred = "oct_id"
    if oct_key_pred not in intv.columns or oct_key_pred not in ext.columns:
        return None, "predictions missing oct_id"
    if oct_key_clin not in clin.columns:
        return None, f"clinical missing column {oct_key_clin}"
    merged = merge_internal_external_clinical(intv, ext, clin, oct_key_pred, oct_key_clin)
    if merged is None or len(merged) == 0:
        return None, "merge yielded 0 rows"
    return merged, "ok"


def build_per_centre_table(
    pred: pd.DataFrame,
    site_col: str,
    thr: float,
    n_boot: int,
    seed: int,
    cohort: str,
) -> pd.DataFrame:
    rows = []
    for site, sub in pred.groupby(site_col, sort=False):
        y = sub["label"].astype(int).values
        pr = sub["prob_pos"].astype(float).values
        boot = bootstrap_auc_ci(y, pr, n_boot=n_boot, seed=seed + hash(str(site)) % 10000)
        m = metrics_at_threshold(y, pr, thr)
        auc_val = float("nan")
        if len(np.unique(y)) >= 2:
            try:
                auc_val = float(roc_auc_score(y, pr))
            except ValueError:
                auc_val = float("nan")
        rows.append(
            {
                "cohort": cohort,
                "centre_id": str(site),
                "n": len(sub),
                "n_positive": int(y.sum()),
                "prevalence": float(y.mean()),
                "threshold": thr,
                "auroc": auc_val,
                "auc_ci_low": boot["auc_ci_low"],
                "auc_ci_high": boot["auc_ci_high"],
                **m,
            }
        )
    return pd.DataFrame(rows)


def try_clinical_subgroups(
    pred: pd.DataFrame,
    clinical: pd.DataFrame,
    oct_key_pred: str,
    oct_key_clin: str,
    out_dir: Path,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    clin = clinical.copy()
    clin = clin.rename(columns={oct_key_clin: "_oct"})
    merged = pred.merge(clin, left_on=oct_key_pred, right_on="_oct", how="inner")
    report: dict[str, Any] = {
        "n_pred": len(pred),
        "n_merged": len(merged),
        "merge_rate": len(merged) / max(len(pred), 1),
    }
    if len(merged) < 10:
        report["subgroup_skipped"] = "merged rows too few"
        (out_dir / "exp3_subgroup_merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    if "AGE" in merged.columns:
        med = merged["AGE"].median()
        merged["age_group"] = np.where(merged["AGE"] >= med, f"age>={med:g}", f"age<{med:g}")
    if "HPV清洗" in merged.columns:
        merged["hpv_16_18"] = merged["HPV清洗"].map(_hpv_16_18)
    if "TCT清洗" in merged.columns:
        merged["tct_group"] = merged["TCT清洗"].map(_tct_binary)

    sub_rows = []
    for col, name in [
        ("hpv_16_18", "HPV 16/18 vs other"),
        ("age_group", "Age median split"),
        ("tct_group", "TCT NILM vs abnormal"),
    ]:
        if col not in merged.columns:
            continue
        for g, sub in merged.groupby(col):
            if g is None or (isinstance(g, float) and np.isnan(g)):
                continue
            y = sub["label"].astype(int).values
            pr = sub["prob_pos"].astype(float).values
            if len(sub) < 5 or len(np.unique(y)) < 2:
                continue
            b = bootstrap_auc_ci(y, pr, n_boot=n_boot, seed=seed)
            sub_rows.append(
                {
                    "subgroup_axis": name,
                    "subgroup_level": str(g),
                    "n": len(sub),
                    "auroc": b["auc"],
                    "auc_ci_low": b["auc_ci_low"],
                    "auc_ci_high": b["auc_ci_high"],
                }
            )
    pd.DataFrame(sub_rows).to_csv(out_dir / "table_exp3_subgroup_external.csv", index=False)
    report["subgroup_rows"] = sub_rows

    if sub_rows:
        df = pd.DataFrame(sub_rows)
        fig, ax = plt.subplots(figsize=(6.5, max(3, 0.35 * len(df) + 1)))
        y = np.arange(len(df))
        ax.barh(y, df["auroc"], height=0.65, color=JOURNAL_PALETTE[0], alpha=0.88, edgecolor="white", linewidth=0.5)
        err_lo = df["auroc"] - df["auc_ci_low"]
        err_hi = df["auc_ci_high"] - df["auroc"]
        ax.errorbar(df["auroc"], y, xerr=[err_lo, err_hi], fmt="none", ecolor="0.3", capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.subgroup_axis}: {r.subgroup_level}" for r in df.itertuples()])
        ax.set_xlabel("AUROC")
        ax.set_xlim(0, 1.05)
        ax.set_title("Subgroup AUROC (external, merged clinical rows)")
        fig.tight_layout()
        save_both(fig, out_dir / "manuscript_figS_subgroup_forest_external")
    (out_dir / "exp3_subgroup_merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oct_root", type=str, default=None)
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--out_dir", type=str, default="logs/manuscript_exp3")
    parser.add_argument("--figures_dir", type=str, default="logs/figures_publication")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "internal_youden"],
        help="fixed: 使用 --threshold；internal_youden: 在内部验证集上选最优 Youden 阈值后应用到内外部",
    )
    parser.add_argument("--n_boot", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="",
        help="含 OCT 或 oct_id 与 AGE/HPV清洗/TCT清洗 等的临床表；需与外部 oct_id 同一命名空间",
    )
    parser.add_argument("--clinical_oct_col", type=str, default="OCT")
    parser.add_argument(
        "--per_centre_tables",
        action="store_true",
        help="额外输出逐中心 CSV/LaTeX（不生成逐医院图）",
    )
    args = parser.parse_args()

    root = Path(args.oct_root) if args.oct_root else Path(__file__).resolve().parents[1]
    logs = root / args.logs_dir
    out = root / args.out_dir
    figs = root / args.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    apply_journal_style()

    ext = pd.read_csv(logs / "external_predictions.csv", usecols=["oct_id", "label", "center_id_external", "prob_pos"])
    ext = ext.rename(columns={"center_id_external": "site"})
    intv = pd.read_csv(logs / "internal_val_predictions.csv", usecols=["oct_id", "label", "center_id", "prob_pos"])
    intv = intv.rename(columns={"center_id": "site"})
    thr = choose_threshold_from_internal(
        intv["label"].astype(int).values,
        intv["prob_pos"].astype(float).values,
        strategy=args.threshold_strategy,
        fallback=args.threshold,
    )

    row_int = summarize_cohort_overall(intv, thr, args.n_boot, args.seed + 99, "internal_validation")
    row_ext = summarize_cohort_overall(ext, thr, args.n_boot, args.seed, "external_test")
    tab_overall = pd.DataFrame([row_int, row_ext])
    tab_overall.to_csv(out / "table_exp3_internal_external_overall.csv", index=False)

    def _tex_escape(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%")

    lines_ov = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Cohort & n & AUROC & 95\% CI & Sens & Spec & Youden J \\",
        r"\hline",
    ]
    cohort_disp = {"internal_validation": "Internal validation", "external_test": "External test"}
    for r in tab_overall.itertuples():
        disp = cohort_disp.get(str(r.cohort), str(r.cohort))
        lines_ov.append(
            f"{_tex_escape(disp)} & {r.n} & {r.auroc:.3f} & [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}] & "
            f"{r.sensitivity:.3f} & {r.specificity:.3f} & {r.youden_j:.3f} \\\\"
        )
    lines_ov.extend([r"\hline", r"\end{tabular}"])
    (out / "table_exp3_internal_external_overall.tex").write_text("\n".join(lines_ov), encoding="utf-8")

    plot_internal_vs_external(
        tab_overall,
        figs / "manuscript_fig_exp3_internal_vs_external",
        "OCT+AI: internal validation vs external test",
    )

    plot_exp3_distribution_figs(
        intv,
        ext,
        figs,
        n_boot=args.n_boot,
        seed_int=args.seed + 99,
        seed_ext=args.seed,
        title_prefix="OCT+AI",
    )

    plot_joint_prob_outcome_marginal(
        build_pred_long_df(intv, ext),
        figs / "manuscript_fig_exp3_joint_prob_outcome_marginal",
        "OCT+AI",
        seed=args.seed + 7,
    )

    if args.per_centre_tables:
        tab_ext = build_per_centre_table(ext, "site", thr, args.n_boot, args.seed, "external_test")
        tab_ext.to_csv(out / "table_exp3_per_centre_external.csv", index=False)
        tab_int = build_per_centre_table(intv, "site", thr, args.n_boot, args.seed + 99, "internal_validation")
        tab_int.to_csv(out / "table_exp3_per_centre_internal_val.csv", index=False)
        lines_pc = [
            r"\begin{tabular}{lrrrrrr}",
            r"\hline",
            r"Centre & n & AUROC & 95\% CI & Sens & Spec & Youden J \\",
            r"\hline",
        ]
        for r in tab_ext.itertuples():
            lines_pc.append(
                f"{_tex_escape(r.centre_id)} & {r.n} & {r.auroc:.3f} & [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}] & "
                f"{r.sensitivity:.3f} & {r.specificity:.3f} & {r.youden_j:.3f} \\\\"
            )
        lines_pc.extend([r"\hline", r"\end{tabular}"])
        (out / "table_exp3_per_centre_external.tex").write_text("\n".join(lines_pc), encoding="utf-8")

    # 亚组
    clin_path = Path(args.clinical_csv) if args.clinical_csv else None
    if clin_path and clin_path.is_file():
        clin = pd.read_csv(clin_path, low_memory=False)
        try_clinical_subgroups(
            ext,
            clin,
            "oct_id",
            args.clinical_oct_col,
            out,
            args.n_boot,
            args.seed,
        )
        merged_ie, imsg = build_merged_ie_from_clinical(intv, ext, clin_path, args.clinical_oct_col)
        report_path = out / "exp3_subgroup_merge_report.json"
        if report_path.exists():
            rep_ie: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            rep_ie = {}
        rep_ie["merge_internal_external_ie"] = imsg
        if merged_ie is not None:
            rep_ie["merged_ie_n"] = len(merged_ie)
            if len(merged_ie) >= 12:
                rep_ie["cohort_subgroup_violin_figs"] = plot_cohort_x_subgroup_split_violins(
                    merged_ie, figs, "OCT+AI"
                )
        report_path.write_text(json.dumps(rep_ie, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        report = {
            "clinical_csv": None,
            "note": "未提供 clinical_csv 或未匹配：当前外部 oct_id（如 M0008_*）与 leave_centers_out 主表 OCT（M221*）命名空间不一致，"
            "需提供覆盖外部 902 例的临床宽表后再跑亚组。",
        }
        (out / "exp3_subgroup_merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "threshold_strategy": args.threshold_strategy,
        "threshold_argument": float(args.threshold),
        "threshold_operating_point": thr,
        "metrics_note": "Youden J = Sensitivity + Specificity - 1 at fixed threshold; LR+/LR- = likelihood ratios.",
        "loho_note": "Leave-one-hospital-out 或 6 家全外部需重新 prepare 数据集并训练；本输出基于当前 loc5out 划分。",
    }
    (out / "exp3_manuscript_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[saved] {out / 'table_exp3_internal_external_overall.csv'}")
    print(f"[saved] {out / 'table_exp3_internal_external_overall.tex'}")
    print(f"[saved] {figs / 'manuscript_fig_exp3_internal_vs_external.png'}")
    print(f"[saved] {figs / 'manuscript_fig_exp3_violin_prob_split_outcome.png'} (+ kde / raincloud / bootstrap violin)")
    print(f"[saved] {figs / 'manuscript_fig_exp3_joint_prob_outcome_marginal.png'} (joint + marginal KDE)")
    if args.per_centre_tables:
        print(f"[saved] {out / 'table_exp3_per_centre_external.csv'} (per-centre)")
        print(f"[saved] {out / 'table_exp3_per_centre_internal_val.csv'} (per-centre)")
    print("[done]")


if __name__ == "__main__":
    main()
