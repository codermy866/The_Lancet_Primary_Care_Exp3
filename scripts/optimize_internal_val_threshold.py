#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score


def metrics_at_threshold(y_true, y_score, thresh):
    y_pred = (y_score >= thresh).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((tp + tn) / len(y_true))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "thresh": float(thresh),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "specificity": specificity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", default="logs/internal_val_predictions.csv")
    parser.add_argument("--out_json", default="logs/internal_threshold_optimization.json")
    parser.add_argument("--out_dir", default="logs")
    parser.add_argument("--recall_target", type=float, default=0.8)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)

    # 找到概率列
    prob_col = None
    for c in ["prob_pos", "prob", "probability", "score"]:
        if c in df.columns:
            prob_col = c
            break
    if prob_col is None:
        for c in df.columns:
            if "prob" in c.lower() or "score" in c.lower():
                prob_col = c
                break
    if prob_col is None:
        raise SystemExit("没有找到概率列 (prob_pos/prob/score 等) 在 %s" % args.pred_csv)
    if "label" not in df.columns:
        raise SystemExit("没有找到 label 列 在 %s" % args.pred_csv)

    y_true = df["label"].astype(int).to_numpy()
    y_score = df[prob_col].astype(float).to_numpy()

    roc_auc = float(roc_auc_score(y_true, y_score))
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recalls, precisions))

    thresholds = np.linspace(0.0, 1.0, 1001)
    rows = []
    best_f1 = -1.0
    best_f1_t = None
    best_youden = -1.0
    best_youden_t = None

    for t in thresholds:
        m = metrics_at_threshold(y_true, y_score, t)
        rows.append(m)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_f1_t = t
        youden = m["recall"] + m["specificity"] - 1.0
        if youden > best_youden:
            best_youden = youden
            best_youden_t = t

    best_f1_metrics = next(r for r in rows if r["thresh"] == best_f1_t)
    best_youden_metrics = next(r for r in rows if r["thresh"] == best_youden_t)

    # 找到满足 recall_target 的最小阈值（使召回 >= 目标）
    t_for_recall = None
    for r in sorted(rows, key=lambda x: x["thresh"]):
        if r["recall"] >= args.recall_target:
            t_for_recall = r["thresh"]
            break

    res = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": {"threshold": best_f1_t, **best_f1_metrics},
        "best_youden": {"threshold": best_youden_t, **best_youden_metrics},
        "recall_target": {"target": args.recall_target, "threshold": t_for_recall},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 保存按 best_f1 的新预测 CSV
    df_out = df.copy()
    df_out["pred_best_f1"] = (df_out[prob_col] >= best_f1_t).astype(int)
    out_csv = os.path.join(args.out_dir, "internal_val_predictions_bestf1.csv")
    df_out.to_csv(out_csv, index=False)

    print("完成：ROC AUC=", roc_auc, "PR AUC=", pr_auc)
    print("best_f1 threshold=", best_f1_t)
    print("best_youden threshold=", best_youden_t)
    print("recall target threshold=", t_for_recall)


if __name__ == "__main__":
    main()
