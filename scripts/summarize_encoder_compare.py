#!/usr/bin/env python3
"""
从 encoder 对照实验目录汇总各子运行的 best / last 验证指标。
用法:
  python scripts/summarize_encoder_compare.py /path/to/encoder_compare_YYYYMMDD_HHMMSS
  python scripts/summarize_encoder_compare.py --runs cnn=/path/a vit_pt=/path/b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _latest_metrics_csv(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None
    csvs = sorted(log_dir.glob("metrics_history_*.csv"))
    return csvs[-1] if csvs else None


def _summarize_run(name: str, log_dir: Path) -> dict:
    csv_path = _latest_metrics_csv(log_dir)
    out: dict = {"run_name": name, "log_dir": str(log_dir.resolve())}
    if csv_path is None:
        for k in (
            "best_epoch",
            "last_epoch",
            "best_val_auc",
            "last_val_auc",
            "error",
        ):
            out[k] = None
        out["error"] = f"no metrics_history_*.csv under {log_dir}"
        return out

    df = pd.read_csv(csv_path)
    if df.empty:
        out["error"] = "empty csv"
        return out

    best_idx = int(df["val_auc"].idxmax())
    last_idx = int(df.index[-1])
    best = df.loc[best_idx]
    last = df.loc[last_idx]

    metric_cols = [c for c in df.columns if c.startswith("val_") or c == "epoch"]
    out["metrics_csv"] = str(csv_path.resolve())
    out["best_epoch"] = int(best["epoch"])
    out["last_epoch"] = int(last["epoch"])
    for c in metric_cols:
        if c == "epoch":
            continue
        out[f"best_{c}"] = float(best[c]) if pd.notna(best[c]) else None
        out[f"last_{c}"] = float(last[c]) if pd.notna(last[c]) else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "compare_root",
        nargs="?",
        default="",
        help="run_encoder_compare.sh 生成的根目录（其下含 cnn/、vit_pt/ 等子目录）",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help='可选：显式指定 name=path，如 cnn=logs/foo/cnn vit_pt=logs/foo/vit_pt"',
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="汇总表输出路径（默认：compare_root/encoder_compare_summary.csv）",
    )
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = []
    if args.runs:
        for item in args.runs:
            if "=" not in item:
                raise SystemExit(f"bad --runs entry (need name=path): {item}")
            name, p = item.split("=", 1)
            runs.append((name.strip(), Path(p.strip())))
    else:
        root = Path(args.compare_root)
        if not root.is_dir():
            raise SystemExit(f"not a directory: {root}")
        default_pairs = [
            ("cnn", root / "cnn"),
            ("vit_pt", root / "vit_pt"),
        ]
        for name, p in default_pairs:
            if p.is_dir():
                runs.append((name, p))

    if not runs:
        raise SystemExit("no runs to summarize")

    rows = []
    for name, p in runs:
        rows.append(_summarize_run(name, p))

    summary = pd.DataFrame(rows)
    if args.out_csv:
        out_path = Path(args.out_csv)
    elif args.compare_root:
        out_path = Path(args.compare_root) / "encoder_compare_summary.csv"
    else:
        out_path = Path("encoder_compare_summary.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding="utf-8")

    # 控制台：挑关键列打印
    key_cols = [
        "run_name",
        "best_epoch",
        "best_val_auc",
        "best_val_pr_auc",
        "best_val_f1",
        "best_val_balanced_acc",
        "best_val_mcc",
        "last_val_auc",
        "last_val_f1",
        "metrics_csv",
    ]
    present = [c for c in key_cols if c in summary.columns]
    print(summary[present].to_string(index=False))
    print(f"\n[saved] {out_path.resolve()}")


if __name__ == "__main__":
    main()
