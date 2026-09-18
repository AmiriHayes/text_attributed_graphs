#!/usr/bin/env python3
"""Full 11-proxy Level1/2/3 correlation tables + proxy-proxy correlation
matrix, for History/Electronics/Toys, reusing already-computed
proxy_list1_{ds}.csv / {ds}_m1_rawgnn_lookup.csv / ragas_results_{ds}.csv."""
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = ['history', 'electronics', 'toys']

PROXIES = ['modularity', 'link_pred_auc', 'n_communities', 'mean_degree',
           'mean_community_size', 'homophily', 'pagerank_std',
           'largest_cc_fraction', 'clustering_coeff_mean',
           'isolated_node_fraction', 'singleton_fraction']


def corr(df, xcol, ycol):
    sub = df[[xcol, ycol]].dropna()
    n = len(sub)
    if n < 3 or sub[xcol].nunique() < 2:
        return float('nan'), float('nan'), n
    rho, p = spearmanr(sub[xcol], sub[ycol])
    return rho, p, n


if __name__ == '__main__':
    for ds in DATASETS:
        ragas = pd.read_csv(REPO_ROOT / f'data/graphrag/ragas_results_{ds}.csv')
        target = ragas.groupby('system_name')['composite_score'].mean().reset_index()
        target['variant'] = target['system_name'].str.replace('_text_based', '', regex=False)
        target = target[['variant', 'composite_score']].rename(columns={'composite_score': 'mean_ragas_composite'})

        l1 = pd.read_csv(REPO_ROOT / f'data/graphrag/proxy_list1_{ds}.csv')
        rawgnn = pd.read_csv(REPO_ROOT / f'data/graphrag/{ds}_m1_rawgnn_lookup.csv')
        merged = target.merge(l1, on='variant', how='inner').merge(rawgnn, on='variant', how='left')
        merged['edge_idx'] = merged['variant'].str.split('_').str[1]
        no_e11b = merged[merged['edge_idx'] != 'E11b']
        underpowered = len(merged) < 5

        print(f'\n{"="*70}\n{ds.upper()}  (full n={len(merged)}, no_e11b n={len(no_e11b)}){"  [UNDERPOWERED]" if underpowered else ""}\n{"="*70}')

        rows = []
        for proxy in PROXIES:
            rf, pf, nf = corr(merged, proxy, 'mean_ragas_composite')
            r0, p0, n0 = corr(no_e11b, proxy, 'mean_ragas_composite')
            rg, pg, ng = corr(no_e11b, proxy, 'raw_gnn_train_mean')
            ragas_independent = bool((not pd.isna(r0)) and (not pd.isna(rg)) and abs(r0) > 0.3 and p0 < 0.1 and abs(rg) < 0.3)
            rows.append({
                'proxy_score': proxy, 'rho_full': rf, 'p_full': pf, 'n_full': nf,
                'rho_no_e11b': r0, 'p_no_e11b': p0, 'n_no_e11b': n0,
                'rho_vs_rawgnn': rg, 'p_vs_rawgnn': pg, 'n_rawgnn': ng,
                'ragas_independent': ragas_independent, 'underpowered': underpowered,
            })
        res = pd.DataFrame(rows)
        res['abs_rho'] = res['rho_no_e11b'].abs()
        res = res.sort_values('abs_rho', ascending=False).drop(columns='abs_rho')
        out = REPO_ROOT / f'data/graphrag/proxy_correlation_{ds}.csv'
        res.to_csv(out, index=False)
        pd.set_option('display.width', 200)
        print(res.to_string(index=False))
        print(f'saved -> {out.name}')

        print(f'\n--- flagged ragas_independent=True ---')
        flagged = res[res['ragas_independent']]
        print(flagged.to_string(index=False) if len(flagged) else '(none)')

        # ── proxy-proxy correlation matrix (no_e11b set) ────────────────
        cols = ['mean_ragas_composite'] + PROXIES
        avail = [c for c in cols if c in no_e11b.columns and no_e11b[c].notna().sum() >= 3]
        cmat = no_e11b[avail].corr(method='spearman')
        fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
        im = ax.imshow(cmat.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(avail))); ax.set_yticks(range(len(avail)))
        ax.set_xticklabels(avail, rotation=90, fontsize=7)
        ax.set_yticklabels(avail, fontsize=7)
        for i in range(len(avail)):
            for j in range(len(avail)):
                v = cmat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if abs(v) > 0.6 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.7, label='Spearman rho')
        ax.set_title(f'{ds} proxy-proxy correlation (E11b excluded, n={len(no_e11b)})', fontsize=10)
        plt.tight_layout()
        mpath = REPO_ROOT / f'data/graphrag/proxy_matrix_{ds}_no_e11b.png'
        fig.savefig(mpath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'saved -> {mpath.name}')

        # crude "clustering" check: mean |rho| among proxy-proxy pairs (excluding target row/col)
        proxy_avail = [c for c in avail if c != 'mean_ragas_composite']
        pm = no_e11b[proxy_avail].corr(method='spearman').values
        iu = np.triu_indices_from(pm, k=1)
        mean_abs_offdiag = float(np.nanmean(np.abs(pm[iu])))
        print(f'mean |rho| among proxy-proxy pairs (E11b excluded): {mean_abs_offdiag:.3f}')
