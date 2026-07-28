#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    matthews_corrcoef,
)


def find_prob_col(df):
    for c in ["prob_pos", "prob", "probability", "score"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "prob" in c.lower() or "score" in c.lower():
            return c
    raise RuntimeError("找不到概率列 (prob_pos/prob/score) 在 DataFrame 中")


def logit(p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1.0 - p))


def pr_auc(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(auc(r, p))


def eval_probs(y_true, y_score, thresh=0.5):
    y_pred = (y_score >= thresh).astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((tp + tn) / n)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = float("nan")
    try:
        pra = pr_auc(y_true, y_score)
    except Exception:
        pra = float("nan")
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")
    try:
        brier = float(brier_score_loss(y_true, y_score))
    except Exception:
        brier = float("nan")
    return {
        "n": int(n),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "specificity": specificity,
        "roc_auc": roc,
        "pr_auc": pra,
        "mcc": mcc,
        "brier": brier,
    }


def calibration_slope_intercept(y_true, y_prob):
    # logistic regression of y on logit(pred)
    x = logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    model.fit(x, y_true)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return slope, intercept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", default="logs/internal_val_predictions.csv")
    parser.add_argument("--external", default="logs/external_predictions.csv")
    parser.add_argument("--threshold_json", default="logs/internal_threshold_optimization.json")
    parser.add_argument("--out_dir", default="logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_int = pd.read_csv(args.internal)
    df_ext = pd.read_csv(args.external)

    prob_int_col = find_prob_col(df_int)
    prob_ext_col = find_prob_col(df_ext)

    y_int = df_int["label"].astype(int).to_numpy()
    p_int = df_int[prob_int_col].astype(float).to_numpy()

    y_ext = df_ext["label"].astype(int).to_numpy()
    p_ext = df_ext[prob_ext_col].astype(float).to_numpy()

    # load best threshold if available
    best_t = None
    if os.path.exists(args.threshold_json):
        with open(args.threshold_json, 'r') as f:
            tj = json.load(f)
            best_t = tj.get('best_f1', {}).get('threshold', None)
    if best_t is None:
        best_t = 0.5

    results = {}

    # external uncalibrated eval at default
    results['external_default_0.5'] = eval_probs(y_ext, p_ext, thresh=0.5)
    results['external_best_threshold'] = eval_probs(y_ext, p_ext, thresh=float(best_t))
    results['external_overall'] = {
        'roc_auc': float(roc_auc_score(y_ext, p_ext)) if len(np.unique(y_ext))>1 else float('nan'),
        'pr_auc': pr_auc(y_ext, p_ext),
        'n': int(len(y_ext)),
        'positive_rate': float(y_ext.mean()),
    }

    # calibration slope/intercept before
    try:
        s_before, i_before = calibration_slope_intercept(y_ext, p_ext)
    except Exception:
        s_before, i_before = float('nan'), float('nan')
    results['external_calibration_before'] = {'slope': s_before, 'intercept': i_before}

    # Platt scaling (logistic regression) trained on internal
    try:
        X_int = logit(p_int).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
        platt.fit(X_int, y_int)
        X_ext = logit(p_ext).reshape(-1, 1)
        p_ext_platt = platt.predict_proba(X_ext)[:, 1]
        df_ext['prob_platt'] = p_ext_platt
        results['platt'] = {
            'coef': float(platt.coef_[0][0]),
            'intercept': float(platt.intercept_[0]),
        }
        results['external_platt_eval'] = eval_probs(y_ext, p_ext_platt, thresh=float(best_t))
        try:
            s_platt, i_platt = calibration_slope_intercept(y_ext, p_ext_platt)
        except Exception:
            s_platt, i_platt = float('nan'), float('nan')
        results['external_calibration_platt'] = {'slope': s_platt, 'intercept': i_platt}
    except Exception as e:
        results['platt_error'] = str(e)

    # Isotonic calibration
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_int, y_int)
        p_ext_iso = iso.predict(p_ext)
        df_ext['prob_isotonic'] = p_ext_iso
        results['external_isotonic_eval'] = eval_probs(y_ext, p_ext_iso, thresh=float(best_t))
        try:
            s_iso, i_iso = calibration_slope_intercept(y_ext, p_ext_iso)
        except Exception:
            s_iso, i_iso = float('nan'), float('nan')
        results['external_calibration_isotonic'] = {'slope': s_iso, 'intercept': i_iso}
    except Exception as e:
        results['isotonic_error'] = str(e)

    # save modified external csv with new probs and preds
    df_ext['pred_default_0.5'] = (df_ext[prob_ext_col] >= 0.5).astype(int)
    df_ext['pred_best_f1'] = (df_ext[prob_ext_col] >= float(best_t)).astype(int)
    if 'prob_platt' in df_ext.columns:
        df_ext['pred_platt_bestf1'] = (df_ext['prob_platt'] >= float(best_t)).astype(int)
    if 'prob_isotonic' in df_ext.columns:
        df_ext['pred_isotonic_bestf1'] = (df_ext['prob_isotonic'] >= float(best_t)).astype(int)

    out_csv = os.path.join(args.out_dir, 'external_predictions_with_calibrated.csv')
    df_ext.to_csv(out_csv, index=False)

    # write results json
    out_json = os.path.join(args.out_dir, 'external_eval_calibration.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('Done. Results written to', out_json)
    print('External default (0.5):', results['external_default_0.5'])
    print('External best_threshold({}):'.format(best_t), results['external_best_threshold'])
    if 'external_platt_eval' in results:
        print('External after Platt (best_thresh):', results['external_platt_eval'])
    if 'external_isotonic_eval' in results:
        print('External after Isotonic (best_thresh):', results['external_isotonic_eval'])


if __name__ == '__main__':
    main()
