"""
统计补充：校准（斜率/截距）、Brier、Bootstrap AUROC 95% CI、决策曲线（DCA）、森林图。

面向多中心泛化实验的补充分析；输出 JSON/CSV + 顶刊风格图（PNG+PDF）。

在 experiments/OCT_traige 下运行:
  python scripts/statistical_supplement_loc5out.py
  python scripts/statistical_supplement_loc5out.py --n_boot 2000 --out_dir logs/figures_publication

输出:
  logs/supplement_statistical_summary.json
  logs/supplement_external_overall.csv, supplement_internal_val_overall.csv
  logs/supplement_external_per_center_stats.csv, supplement_internal_val_per_center_stats.csv
  figures_publication/pub_fig09_forest_auroc_bootstrap.* (外部)
  figures_publication/pub_fig12_forest_auroc_internal.* (内部验证)
  figures_publication/pub_fig10_decision_curve_external.*, pub_fig11_decision_curve_internal.*
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_pub", _SCRIPT_DIR / "publication_figures_loc5out.py")
_pub = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_pub)
apply_journal_style = _pub.apply_journal_style
save_both = _pub.save_both

# 与 generate_lancet_latex_tables 对齐的英文中心名（作图用）
_SITE_EN: dict[str, str] = {
    "十堰市人民医院": "Shiyan People's Hospital",
    "恩施州中心医院": "Enshi Prefecture Central Hospital",
    "武大人民医院": "Renmin Hospital of Wuhan University",
    "荆州市第一人民医院": "Jingzhou First People's Hospital",
    "襄阳市中心医院": "Xiangyang Central Hospital",
    "5c_unknown_center": "Unmapped site",
    "AnYang": "Anyang",
    "Hua_Xi": "Huaxi",
    "HuaXi": "Huaxi",
    "liaoning": "Liaoning",
    "ZhengDaSanFu": "Zhengda Sanfu",
}


def _site_en(s: str) -> str:
    s = str(s).strip()
    return _SITE_EN.get(s, s)


def calibration_slope_intercept(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """对 logit(预测概率) 拟合 Logistic(Y)，得到校准斜率/截距（常见报告方式）。"""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    X = logit(prob).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        lr.fit(X, y_true)
    return {
        "calibration_slope": float(lr.coef_[0, 0]),
        "calibration_intercept": float(lr.intercept_[0]),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "auc_ci_low": float("nan"), "auc_ci_high": float("nan"), "n_boot_valid": 0}
    auc_obs = roc_auc_score(y_true, prob)
    rng = np.random.default_rng(seed)
    store: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        store.append(roc_auc_score(yt, prob[idx]))
    if not store:
        return {
            "auc": float(auc_obs),
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "n_boot_valid": 0,
        }
    lo, hi = np.percentile(store, [2.5, 97.5])
    return {
        "auc": float(auc_obs),
        "auc_ci_low": float(lo),
        "auc_ci_high": float(hi),
        "n_boot_valid": len(store),
    }


def decision_curve_data(
    y_true: np.ndarray,
    prob: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (thresholds, nb_model, nb_all, nb_none)。Vickers & Elkin 形式。"""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n = len(y_true)
    pi = y_true.mean()
    nb_model = []
    nb_all = []
    nb_none = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            nb_model.append(np.nan)
            nb_all.append(np.nan)
            nb_none.append(0.0)
            continue
        pred = prob >= pt
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        ratio = pt / (1.0 - pt)
        nb_model.append((tp / n) - (fp / n) * ratio)
        # Treat all: 全部判为阳性
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        nb_all.append((tp_all / n) - (fp_all / n) * ratio)
        nb_none.append(0.0)
    return thresholds, np.array(nb_model), np.array(nb_all), np.array(nb_none)


def fig_forest_auroc(rows: list[dict[str, Any]], out: Path, title: str, stem: str) -> None:
    """rows: site, auc, ci_low, ci_high, n, note(optional)"""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("auc", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.8, max(3.2, 0.42 * len(df) + 1.2)))
    aucs = df["auc"].values.astype(float)
    lo = df["ci_low"].values.astype(float)
    hi = df["ci_high"].values.astype(float)
    err = np.array([aucs - lo, hi - aucs])
    colors = ["#999999" if (row.get("n", 0) or 0) < 30 else "#0072B2" for row in df.to_dict("records")]
    ax.errorbar(aucs, y, xerr=err, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=2)
    ax.scatter(aucs, y, c=colors, s=42, zorder=3, edgecolors="white", linewidths=0.6)
    ax.axvline(0.5, color="#BBBBBB", ls="--", lw=0.9, zorder=0)
    labels = [_site_en(str(x)) for x in df["site"].values]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC (95% CI, bootstrap)")
    ax.set_xlim(0.0, 1.02)
    ax.set_title(title)
    for i, r in enumerate(df.itertuples()):
        n = int(r.n) if pd.notna(r.n) else 0
        ax.text(1.01, i, f"n={n}", va="center", fontsize=7, color="0.4")
    fig.tight_layout()
    save_both(fig, out / stem)


def fig_decision_curve(
    y_true: np.ndarray,
    prob: np.ndarray,
    out: Path,
    title: str,
    stem: str,
) -> None:
    th = np.linspace(0.01, 0.99, 99)
    _, nb_m, nb_a, nb_n = decision_curve_data(y_true, prob, th)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(th, nb_m, color="#0072B2", lw=2, label="Model")
    ax.plot(th, nb_a, color="#888888", ls="--", lw=1.2, label="Treat all")
    ax.plot(th, nb_n, color="#333333", ls=":", lw=1.2, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1)
    vm = float(np.nanmax(nb_m[np.isfinite(nb_m)])) if np.any(np.isfinite(nb_m)) else 0.15
    vn = float(np.nanmin(nb_m[np.isfinite(nb_m)])) if np.any(np.isfinite(nb_m)) else -0.05
    pad = 0.05 * max(vm - vn, 0.05)
    ax.set_ylim(vn - pad, vm + pad)
    fig.tight_layout()
    save_both(fig, out / stem)


def _load_external(logs: Path) -> pd.DataFrame:
    p = logs / "external_predictions.csv"
    df = pd.read_csv(p, encoding="utf-8", usecols=["label", "center_id_external", "prob_pos"])
    df = df.rename(columns={"center_id_external": "site"})
    return df


def _load_internal_val(logs: Path) -> pd.DataFrame:
    p = logs / "internal_val_predictions.csv"
    df = pd.read_csv(p, encoding="utf-8", usecols=["label", "center_id", "prob_pos"])
    df = df.rename(columns={"center_id": "site"})
    return df


def run_external(
    logs: Path,
    out_fig: Path,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    df = _load_external(logs)
    y = df["label"].astype(int).values
    p = df["prob_pos"].astype(float).values

    summary: dict[str, Any] = {
        "cohort": "external_test",
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, p)),
    }
    summary.update(calibration_slope_intercept(y, p))

    boot = bootstrap_auc_ci(y, p, n_boot=n_boot, seed=seed)
    summary["auroc"] = boot

    per_rows: list[dict[str, Any]] = []
    for site, sub in df.groupby("site", sort=False):
        ys = sub["label"].astype(int).values
        pr = sub["prob_pos"].astype(float).values
        b = bootstrap_auc_ci(ys, pr, n_boot=n_boot, seed=seed + hash(str(site)) % 10000)
        per_rows.append(
            {
                "site": str(site),
                "site_label_en": _site_en(str(site)),
                "n": int(len(sub)),
                "positive_n": int(ys.sum()),
                "auroc": b["auc"],
                "auc_ci_low": b["auc_ci_low"],
                "auc_ci_high": b["auc_ci_high"],
                "n_boot_valid": b["n_boot_valid"],
                "brier": float(brier_score_loss(ys, pr)) if len(np.unique(ys)) >= 1 else float("nan"),
                **(
                    calibration_slope_intercept(ys, pr)
                    if len(np.unique(ys)) >= 2
                    else {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
                ),
            }
        )

    forest_rows = [
        {
            "site": r["site"],
            "auc": r["auroc"],
            "ci_low": r["auc_ci_low"],
            "ci_high": r["auc_ci_high"],
            "n": r["n"],
        }
        for r in per_rows
        if np.isfinite(r["auroc"])
    ]
    fig_forest_auroc(
        forest_rows,
        out_fig,
        title="External test: site-specific AUROC with 95% bootstrap CI",
        stem="pub_fig09_forest_auroc_bootstrap",
    )
    fig_decision_curve(
        y,
        p,
        out_fig,
        title="Decision curve (external test)",
        stem="pub_fig10_decision_curve_external",
    )

    return {"summary": summary, "per_center": per_rows}


def run_internal_val(
    logs: Path,
    out_fig: Path,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    df = _load_internal_val(logs)
    y = df["label"].astype(int).values
    pr = df["prob_pos"].astype(float).values
    fig_decision_curve(
        y,
        pr,
        out_fig,
        title="Decision curve (internal validation)",
        stem="pub_fig11_decision_curve_internal",
    )
    out = {
        "cohort": "internal_validation",
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, pr)),
        **calibration_slope_intercept(y, pr),
        "auroc": bootstrap_auc_ci(y, pr, n_boot=n_boot, seed=seed + 1),
    }
    per_rows = []
    for site, sub in df.groupby("site"):
        ys = sub["label"].astype(int).values
        pv = sub["prob_pos"].astype(float).values
        b = bootstrap_auc_ci(ys, pv, n_boot=n_boot, seed=seed + 2 + hash(str(site)) % 10000)
        per_rows.append(
            {
                "site": str(site),
                "site_label_en": _site_en(str(site)),
                "n": int(len(sub)),
                "positive_n": int(ys.sum()),
                "auroc": b["auc"],
                "auc_ci_low": b["auc_ci_low"],
                "auc_ci_high": b["auc_ci_high"],
                "n_boot_valid": b["n_boot_valid"],
                "brier": float(brier_score_loss(ys, pv)),
                **calibration_slope_intercept(ys, pv),
            }
        )
    forest_rows = [
        {
            "site": r["site"],
            "auc": r["auroc"],
            "ci_low": r["auc_ci_low"],
            "ci_high": r["auc_ci_high"],
            "n": r["n"],
        }
        for r in per_rows
        if np.isfinite(r["auroc"])
    ]
    fig_forest_auroc(
        forest_rows,
        out_fig,
        title="Internal validation: AUROC by centre (95% bootstrap CI)",
        stem="pub_fig12_forest_auroc_internal",
    )
    return {"summary": out, "per_center": per_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oct_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="logs/figures_publication")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--n_boot", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.oct_root) if args.oct_root else Path(__file__).resolve().parents[1]
    logs = root / args.logs_dir
    out_fig = root / args.out_dir
    apply_journal_style()

    ext = run_external(logs, out_fig, n_boot=args.n_boot, seed=args.seed)
    internal = run_internal_val(logs, out_fig, n_boot=args.n_boot, seed=args.seed)

    out_json = logs / "supplement_statistical_summary.json"
    payload = {
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "external": ext,
        "internal_validation": internal,
        "notes": [
            "AUROC CI: percentile bootstrap on patient-level resampling (same cohort size).",
            "Calibration slope/intercept: logistic regression of Y on logit(predicted probability).",
            "DCA: net benefit = TP/N - FP/N * pt/(1-pt) (Vickers & Elkin).",
            "Small centres: CI may be very wide; grey in forest plot marks n<30.",
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame(ext["per_center"]).to_csv(logs / "supplement_external_per_center_stats.csv", index=False)
    pd.DataFrame(internal["per_center"]).to_csv(logs / "supplement_internal_val_per_center_stats.csv", index=False)

    s_ext = ext["summary"]
    pd.DataFrame(
        [
            {
                "cohort": s_ext.get("cohort"),
                "n": s_ext["n"],
                "positive_rate": s_ext["positive_rate"],
                "brier_score": s_ext["brier_score"],
                "calibration_slope": s_ext["calibration_slope"],
                "calibration_intercept": s_ext["calibration_intercept"],
                **s_ext["auroc"],
            }
        ]
    ).to_csv(logs / "supplement_external_overall.csv", index=False)

    s_in = internal["summary"]
    pd.DataFrame(
        [
            {
                "cohort": s_in.get("cohort"),
                "n": s_in["n"],
                "positive_rate": s_in["positive_rate"],
                "brier_score": s_in["brier_score"],
                "calibration_slope": s_in["calibration_slope"],
                "calibration_intercept": s_in["calibration_intercept"],
                **s_in["auroc"],
            }
        ]
    ).to_csv(logs / "supplement_internal_val_overall.csv", index=False)

    print(f"[saved] {out_json}")
    print(f"[saved] {logs / 'supplement_external_per_center_stats.csv'}")
    print(f"[saved] {logs / 'supplement_internal_val_per_center_stats.csv'}")
    print(f"[saved] {out_fig / 'pub_fig09_forest_auroc_bootstrap.png'}")
    print(f"[saved] {out_fig / 'pub_fig10_decision_curve_external.png'}")
    print(f"[saved] {out_fig / 'pub_fig11_decision_curve_internal.png'}")
    print(f"[saved] {out_fig / 'pub_fig12_forest_auroc_internal.png'}")
    print("[done]")


if __name__ == "__main__":
    main()
