#!/usr/bin/env python3
"""Step 2/3/4: fit decision trees on proxy scores (features=N/E/T, same
hyperparams as the GNN pipeline), predict on the training variants, and
check whether those tree-smoothed predictions correlate with actual RAGAS
better than the raw proxy value did. raw_gnn is the baseline tree.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'code'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from decision_tree_analysis import fit_tree
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = ['arxiv', 'amazon', 'history', 'electronics', 'toys']
DS_FILES = {
    'arxiv': 'ragas_results_arxiv_all_variants.csv', 'amazon': 'ragas_results_amazon.csv',
    'history': 'ragas_results_history.csv', 'electronics': 'ragas_results_electronics.csv',
    'toys': 'ragas_results_toys.csv',
}
PROXIES = ['modularity', 'link_pred_auc', 'n_communities', 'mean_degree', 'mean_community_size', 'pagerank_std']


def load_merged(ds: str) -> pd.DataFrame:
    ragas = pd.read_csv(REPO_ROOT / f'data/graphrag/{DS_FILES[ds]}')
    target = ragas.groupby('system_name')['composite_score'].mean().reset_index()
    target['variant'] = target['system_name'].str.replace('_text_based', '', regex=False)
    target = target[['variant', 'composite_score']].rename(columns={'composite_score': 'mean_ragas_composite'})

    l1 = pd.read_csv(REPO_ROOT / f'data/graphrag/proxy_list1_{ds}.csv')
    rawgnn = pd.read_csv(REPO_ROOT / f'data/graphrag/{ds}_m1_rawgnn_lookup.csv')
    merged = target.merge(l1, on='variant', how='inner').merge(rawgnn, on='variant', how='left')

    parts = merged['variant'].str.split('_', expand=True)
    merged['Node_Idx'], merged['Edge_Idx'], merged['Text_Idx'] = parts[0], parts[1], parts[2]
    return merged


def tree_rho(df: pd.DataFrame, target_col: str) -> dict:
    """Fit tree(N,E,T -> target_col), predict on same rows, rho vs RAGAS
    (full set and E11b-excluded)."""
    sub = df.dropna(subset=[target_col, 'mean_ragas_composite']).copy()
    n_full = len(sub)
    if n_full < 5 or sub[target_col].nunique() < 2:
        return {'rho_full': float('nan'), 'p_full': float('nan'), 'n_full': n_full,
                'rho_no_e11b': float('nan'), 'p_no_e11b': float('nan'), 'n_no_e11b': 0,
                'top_split_axis': None, 'importances': None}

    tr = fit_tree(sub, target_col=target_col, features=['Node_Idx', 'Edge_Idx', 'Text_Idx'], max_depth=8)
    preds = tr['tree'].predict(tr['X'])
    sub = sub.reset_index(drop=True)
    sub['tree_pred'] = preds

    rho_f, p_f = spearmanr(sub['tree_pred'], sub['mean_ragas_composite'])

    no_e11b = sub[sub['Edge_Idx'] != 'E11b']
    if len(no_e11b) >= 3 and no_e11b['tree_pred'].nunique() > 1:
        rho_0, p_0 = spearmanr(no_e11b['tree_pred'], no_e11b['mean_ragas_composite'])
    else:
        rho_0, p_0 = float('nan'), float('nan')

    imp = dict(zip(tr['ohe_names'], tr['tree'].feature_importances_))
    axis_imp = {}
    for k, v in imp.items():
        col, _ = tr['ohe_to_axis_val'][k]
        axis_imp[col] = axis_imp.get(col, 0.0) + v
    top_axis = max(axis_imp, key=axis_imp.get) if axis_imp else None

    return {'rho_full': rho_f, 'p_full': p_f, 'n_full': n_full,
            'rho_no_e11b': rho_0, 'p_no_e11b': p_0, 'n_no_e11b': len(no_e11b),
            'top_split_axis': top_axis, 'importances': axis_imp, 'tree_result': tr}


if __name__ == '__main__':
    data = {ds: load_merged(ds) for ds in DATASETS}

    print('=== STEP 1: per-dataset n (already confirmed) ===')
    for ds in DATASETS:
        print(f'  {ds}: n={len(data[ds])}')

    print('\n\n=== STEP 2/3: proxy trees + raw_gnn baseline tree ===')
    results = []
    trees_by_proxy_ds = {}
    for target in PROXIES + ['raw_gnn_train_mean']:
        print(f'\n{"="*70}\nTARGET: {target}\n{"="*70}')
        for ds in DATASETS:
            r = tree_rho(data[ds], target)
            flag = '  [UNDERPOWERED]' if ds == 'history' else ''
            print(f'  {ds:<12} rho_full={r["rho_full"]:+.3f} (p={r["p_full"]:.3f}, n={r["n_full"]})   '
                  f'rho_no_e11b={r["rho_no_e11b"]:+.3f} (p={r["p_no_e11b"]:.3f}, n={r["n_no_e11b"]})   '
                  f'top_split={r["top_split_axis"]}{flag}')
            results.append({
                'proxy': target, 'dataset': ds,
                'rho_full': r['rho_full'], 'p_full': r['p_full'], 'n_full': r['n_full'],
                'rho_no_e11b': r['rho_no_e11b'], 'p_no_e11b': r['p_no_e11b'], 'n_no_e11b': r['n_no_e11b'],
                'top_split_axis': r['top_split_axis'],
                'node_importance': r['importances'].get('Node_Idx', 0) if r['importances'] else None,
                'edge_importance': r['importances'].get('Edge_Idx', 0) if r['importances'] else None,
                'text_importance': r['importances'].get('Text_Idx', 0) if r['importances'] else None,
                'underpowered': ds == 'history',
            })
            if target in PROXIES:
                trees_by_proxy_ds[(target, ds)] = r.get('tree_result')

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPO_ROOT / 'data/graphrag/proxy_tree_validation_all_datasets.csv', index=False)
    print(f'\nsaved -> proxy_tree_validation_all_datasets.csv')

    # ── Step 3 summary table: proxy | per-dataset rho_full | mean_rho ──
    print(f'\n\n{"="*70}\nSTEP 3 SUMMARY TABLE (rho_full, tree-predicted vs actual RAGAS)\n{"="*70}')
    pivot = res_df.pivot(index='proxy', columns='dataset', values='rho_full')[DATASETS]
    pivot['mean_rho'] = pivot[DATASETS].mean(axis=1)
    pivot = pivot.sort_values('mean_rho', ascending=False)
    pd.set_option('display.width', 160)
    print(pivot.to_string())

    print(f'\n\n{"="*70}\nSTEP 3 SUMMARY TABLE (rho_no_e11b, tree-predicted vs actual RAGAS)\n{"="*70}')
    pivot2 = res_df.pivot(index='proxy', columns='dataset', values='rho_no_e11b')[DATASETS]
    pivot2['mean_rho'] = pivot2[DATASETS].mean(axis=1)
    pivot2 = pivot2.sort_values('mean_rho', ascending=False)
    print(pivot2.to_string())

    # ── Step 4: cross-dataset transfer for best proxy (by no_e11b mean_rho, excluding raw_gnn) ──
    best_proxy = pivot2.drop(index='raw_gnn_train_mean', errors='ignore').index[0]
    print(f'\n\n{"="*70}\nSTEP 4: CROSS-DATASET TRANSFER for best proxy = {best_proxy}\n{"="*70}')

    transfer = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
    for train_ds in DATASETS:
        tr = trees_by_proxy_ds.get((best_proxy, train_ds))
        if tr is None:
            continue
        for test_ds in DATASETS:
            test_df = data[test_ds].dropna(subset=[best_proxy, 'mean_ragas_composite']).copy()
            if len(test_df) < 3:
                continue
            X_test = tr['ohe'].transform(test_df[['Node_Idx', 'Edge_Idx', 'Text_Idx']])
            preds = tr['tree'].predict(X_test)
            if len(set(preds)) < 2 or test_df['mean_ragas_composite'].nunique() < 2:
                rho = float('nan')
            else:
                rho, _ = spearmanr(preds, test_df['mean_ragas_composite'])
            transfer.loc[train_ds, test_ds] = rho

    print('rows = tree trained on this dataset\'s proxy tree; columns = applied to')
    print(transfer.round(3).to_string())
    transfer.to_csv(REPO_ROOT / f'data/graphrag/proxy_tree_transfer_{best_proxy}.csv')
    print(f'\nsaved -> proxy_tree_transfer_{best_proxy}.csv')
