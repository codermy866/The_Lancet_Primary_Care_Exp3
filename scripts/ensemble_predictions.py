#!/usr/bin/env python3
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def find_prob_cols(df):
    return [c for c in df.columns if 'prob' in c.lower() or 'score' in c.lower()]


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
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--pattern", default="external_predictions*.csv")
    parser.add_argument("--out_dir", default="logs")
    parser.add_argument("--do_stacking", action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.logs_dir, args.pattern)))
    if not files:
        raise SystemExit('No prediction files found with pattern %s in %s' % (args.pattern, args.logs_dir))

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # require oct_id and label
        if 'label' not in df.columns:
            print('[skip]', f, 'missing label')
            continue
        dfs.append((f, df))

    # build a robust merged DataFrame: prefer external_predictions.csv as base (keep its rows),
    # left-merge other prediction files on `oct_id` when available, and compute mean across available prob cols skipping NaN.
    base_df = None
    for fname, df in dfs:
        if os.path.basename(fname) == 'external_predictions.csv':
            base_df = df.copy()
            break
    if base_df is None:
        base_df = dfs[0][1].copy()

    if 'oct_id' in base_df.columns:
        key = 'oct_id'
        merged = base_df[[key, 'label']].copy()
        merged[key] = merged[key].astype(str)
    else:
        key = None
        merged = base_df[['label']].copy()
        merged['idx'] = merged.index
        key = 'idx'

    # collect prob cols and left-merge them into merged (keep base rows)
    prob_cols = []
    for fname, df in dfs:
        pcols = find_prob_cols(df)
        if not pcols:
            print('[skip]', fname, 'no prob cols')
            continue
        pc = pcols[0]
        colname = os.path.splitext(os.path.basename(fname))[0] + '__' + pc
        tmp = df[[pc]].copy()
        tmp.columns = [colname]
        if key in df.columns:
            tmp[key] = df[key].astype(str)
            merged = merged.merge(tmp, on=key, how='left')
        else:
            # align by index if no key available in this file
            tmp[key] = df.index
            merged = merged.merge(tmp, on=key, how='left')
        prob_cols.append(colname)

    # compute ensemble mean across available probability columns (skip NaN)
    if prob_cols:
        merged['prob_ensemble_mean'] = merged[prob_cols].astype(float).mean(axis=1, skipna=True)
    else:
        merged['prob_ensemble_mean'] = float('nan')

    # evaluate only on rows that have both label and an ensemble probability
    df_eval = merged.dropna(subset=['label', 'prob_ensemble_mean'])
    if len(df_eval) > 0:
        ensemble_eval = eval_probs(df_eval['label'].astype(int).to_numpy(), df_eval['prob_ensemble_mean'].astype(float).to_numpy())
    else:
        ensemble_eval = {'n': 0, 'roc_auc': float('nan'), 'pr_auc': float('nan'), 'positive_rate': float('nan')}

    res = {'files_used': [os.path.basename(f) for f, _ in dfs], 'n_models': len(prob_cols), 'ensemble_mean_eval': ensemble_eval}

    # stacking using internal val if requested
    if args.do_stacking:
        try:
            # look for internal_val_predictions.csv in logs and use it to train meta-learner
            int_path = os.path.join(args.logs_dir, 'internal_val_predictions.csv')
            if os.path.exists(int_path):
                df_int = pd.read_csv(int_path)
                # build features from the same files but internal versions must exist with same base names
                feat_dfs = []
                feature_names = []
                for fname, _ in dfs:
                    base = os.path.splitext(os.path.basename(fname))[0]
                    # try to find internal file with same base
                    cand = os.path.join(args.logs_dir, base.replace('external', 'internal') + '.csv')
                    if not os.path.exists(cand):
                        # fallback to generic internal_val_predictions
                        cand = int_path
                    df_c = pd.read_csv(cand)
                    pcols_c = find_prob_cols(df_c)
                    if not pcols_c:
                        raise ValueError(f'No prob cols found in candidate {cand}')
                    pc = pcols_c[0]
                    feat_name = base + '__' + pc
                    feature_names.append(feat_name)
                    feat = df_c[[pc]].copy()
                    feat.columns = [feat_name]
                    feat_dfs.append(feat)
                # concatenate features horizontally from int_path (assume same order)
                X_meta = pd.concat(feat_dfs, axis=1).astype(float)
                y_meta = df_int['label'].astype(int)
                # simple NaN check and robust imputation (fill per-column mean, fallback to global)
                if X_meta.isna().any().any():
                    col_means = X_meta.mean()
                    global_mean = float(col_means.mean())
                    if np.isnan(global_mean):
                        global_mean = 0.5
                    X_meta = X_meta.fillna(col_means)
                    X_meta = X_meta.fillna(global_mean)
                X_meta_np = X_meta.to_numpy()
                y_meta_np = y_meta.to_numpy()
                # train logistic regression meta
                meta = LogisticRegression(max_iter=1000)
                meta.fit(X_meta_np, y_meta_np)
                # build external features aligned to merged rows using the same key
                ext_feat_list = []
                for fname, _ in dfs:
                    df_e = pd.read_csv(fname)
                    pcols_e = find_prob_cols(df_e)
                    if not pcols_e:
                        raise ValueError(f'No prob cols found in external file {fname}')
                    pc = pcols_e[0]
                    base = os.path.splitext(os.path.basename(fname))[0]
                    feat_name = base + '__' + pc
                    if key in df_e.columns:
                        # reindex by key (as string) to match merged order
                        s = df_e.set_index(key)[pc].astype(float).rename(feat_name)
                        s.index = s.index.astype(str)
                        s = s.reindex(merged[key].astype(str))
                    else:
                        # no key in this file, create NaN column of same length as merged
                        s = pd.Series([float('nan')] * len(merged), index=merged.index, name=feat_name)
                    ext_feat_list.append(s)
                X_ext = pd.concat(ext_feat_list, axis=1).astype(float)
                # robust imputation: fill per-column means, then fallback to a global mean (0.5 if needed)
                if X_ext.isna().any().any():
                    col_means_ext = X_ext.mean()
                    global_mean_ext = float(col_means_ext.mean())
                    if np.isnan(global_mean_ext):
                        global_mean_ext = 0.5
                    X_ext = X_ext.fillna(col_means_ext)
                    X_ext = X_ext.fillna(global_mean_ext)
                X_ext_meta = X_ext.to_numpy()
                p_stack = meta.predict_proba(X_ext_meta)[:, 1]
                merged['prob_stack'] = p_stack
                res['stacking_coef'] = meta.coef_.tolist()
                res['stacking_intercept'] = meta.intercept_.tolist()
                # evaluate on rows with label
                df_stack_eval = merged.dropna(subset=['label'])
                if len(df_stack_eval) > 0:
                    res['ensemble_stack_eval'] = eval_probs(df_stack_eval['label'].astype(int).to_numpy(), p_stack[:len(df_stack_eval)])
                else:
                    res['ensemble_stack_eval'] = {'n': 0, 'roc_auc': float('nan'), 'pr_auc': float('nan'), 'positive_rate': float('nan')}
            else:
                res['stacking_error'] = 'internal_val_predictions.csv not found in logs'
        except Exception as e:
            import traceback
            res['stacking_error'] = str(e)
            res['stacking_traceback'] = traceback.format_exc()

    # save merged and metrics
    out_csv = os.path.join(args.out_dir, 'external_ensemble_predictions.csv')
    merged.to_csv(out_csv, index=False)
    out_json = os.path.join(args.out_dir, 'external_ensemble_eval.json')
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print('Saved ensemble csv:', out_csv)
    print('Saved ensemble json:', out_json)
    print('Ensemble eval:', res['ensemble_mean_eval'])


if __name__ == '__main__':
    main()
