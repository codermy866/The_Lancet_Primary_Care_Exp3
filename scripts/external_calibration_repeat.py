#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def find_prob_col(df):
    for c in ["prob_pos", "prob", "probability", "score"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "prob" in c.lower() or "score" in c.lower():
            return c
    raise RuntimeError("找不到概率列 (prob_pos/prob/score) 在 DataFrame 中")


def pr_auc(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(auc(r, p))


def eval_probs(y_true, y_score):
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = float('nan')
    try:
        pra = pr_auc(y_true, y_score)
    except Exception:
        pra = float('nan')
    return {"n": int(len(y_true)), "roc_auc": roc, "pr_auc": pra, "positive_rate": float(np.mean(y_true))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_csv", default="logs/external_predictions.csv")
    parser.add_argument("--cal_frac", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", default="logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.external_csv)
    prob_col = find_prob_col(df)
    if 'label' not in df.columns:
        raise SystemExit('external csv 缺少 label 列')

    y = df['label'].astype(int).to_numpy()
    p = df[prob_col].astype(float).to_numpy()

    # stratified split
    idx = np.arange(len(df))
    cal_idx, hold_idx = train_test_split(idx, test_size=1-args.cal_frac, stratify=y, random_state=args.random_state)
    df_cal = df.iloc[cal_idx].reset_index(drop=True)
    df_hold = df.iloc[hold_idx].reset_index(drop=True)

    # Save splits
    cal_csv = os.path.join(args.out_dir, 'external_calibration_set.csv')
    hold_csv = os.path.join(args.out_dir, 'external_holdout_set.csv')
    df_cal.to_csv(cal_csv, index=False)
    df_hold.to_csv(hold_csv, index=False)

    res = {'n_total': int(len(df)), 'n_cal': int(len(df_cal)), 'n_hold': int(len(df_hold)), 'prob_col': prob_col}

    # Eval before calibration
    res['hold_before'] = eval_probs(df_hold['label'].astype(int).to_numpy(), df_hold[prob_col].astype(float).to_numpy())

    # Platt on calibration
    try:
        X_cal = np.log(df_cal[prob_col].astype(float).clip(1e-15, 1 - 1e-15) / (1 - df_cal[prob_col].astype(float).clip(1e-15, 1 - 1e-15))).reshape(-1,1)
        platt = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
        platt.fit(X_cal, df_cal['label'].astype(int).to_numpy())
        X_hold = np.log(df_hold[prob_col].astype(float).clip(1e-15, 1 - 1e-15) / (1 - df_hold[prob_col].astype(float).clip(1e-15, 1 - 1e-15))).reshape(-1,1)
        p_hold_platt = platt.predict_proba(X_hold)[:,1]
        df_hold['prob_platt'] = p_hold_platt
        res['platt_coef'] = float(platt.coef_[0][0])
        res['platt_intercept'] = float(platt.intercept_[0])
        res['hold_platt_eval'] = eval_probs(df_hold['label'].astype(int).to_numpy(), p_hold_platt)
    except Exception as e:
        res['platt_error'] = str(e)

    # Isotonic on calibration
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(df_cal[prob_col].astype(float).to_numpy(), df_cal['label'].astype(int).to_numpy())
        p_hold_iso = iso.predict(df_hold[prob_col].astype(float).to_numpy())
        df_hold['prob_isotonic'] = p_hold_iso
        res['hold_isotonic_eval'] = eval_probs(df_hold['label'].astype(int).to_numpy(), p_hold_iso)
    except Exception as e:
        res['isotonic_error'] = str(e)

    # Save holdout with calibrated probs
    out_hold_csv = os.path.join(args.out_dir, 'external_holdout_with_calibrated.csv')
    df_hold.to_csv(out_hold_csv, index=False)

    out_json = os.path.join(args.out_dir, 'external_calibration_repeat.json')
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print('Saved calibration set:', cal_csv)
    print('Saved holdout set:', hold_csv)
    print('Saved holdout with calibrated probs:', out_hold_csv)
    print('Saved summary json:', out_json)


if __name__ == '__main__':
    main()
