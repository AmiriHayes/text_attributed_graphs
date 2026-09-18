#!/usr/bin/env python3
"""Practitioner-framing stress test: 6 graph-only proxies x 5 datasets x
3 correlation levels (full, E11b-excluded, vs raw_gnn), plus a ranking
test (Spearman rank-corr between proxy-induced ranking and actual RAGAS
ranking, plus top-1 precision) compared against raw_gnn as baseline."""
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = ['arxiv', 'amazon', 'history', 'electronics', 'toys']
DS_LABEL = {'arxiv': 'ArXiv', 'amazon': 'Amazon', 'history': 'History',
            'electronics': 'Electronics', 'toys': 'Toys'}

PROXIES = ['modularity', 'link_pred_auc', 'n_communities', 'mean_degree',
           'mean_community_size', 'homophily']
# Direction for ranking: True = higher is better (descending rank), False = lower is better
HIGHER_IS_BETTER = {
    'modularity': True, 'link_pred_auc': True, 'n_communities': True,
    'mean_degree': False, 'mean_community_size': False, 'homophily': True,
}

RAGAS_COL_MAP = {}  # arxiv's file uses list1_arxiv.csv already keyed by 'variant'


def load_dataset(ds: str) -> pd.DataFrame:
    # arxiv's 20-variant sweep lives in a differently-named file; the plain
    # ragas_results_arxiv.csv is the earlier 5-config pilot test, wrong shape.
    fname = 'ragas_results_arxiv_all_variants.csv' if ds == 'arxiv' else f'ragas_results_{ds}.csv'
    ragas_file = REPO_ROOT / f'data/graphrag/{fname}'
    ragas = pd.read_csv(ragas_file)
    target = ragas.groupby('system_name')['composite_score'].mean().reset_index()
    target['variant'] = target['system_name'].str.replace('_text_based', '', regex=False)
    target = target[['variant', 'composite_score']].rename(columns={'composite_score': 'mean_ragas_composite'})

    l1 = pd.read_csv(REPO_ROOT / f'data/graphrag/proxy_list1_{ds}.csv')
    rawgnn = pd.read_csv(REPO_ROOT / f'data/graphrag/{ds}_m1_rawgnn_lookup.csv')

    merged = target.merge(l1, on='variant', how='inner').merge(rawgnn, on='variant', how='left')
    merged['edge_idx'] = merged['variant'].str.split('_').str[1]
    merged['dataset'] = ds
    return merged


def corr(df, xcol, ycol):
    sub = df[[xcol, ycol]].dropna()
    n = len(sub)
    if n < 3 or sub[xcol].nunique() < 2:
        return float('nan'), float('nan'), n
    rho, p = spearmanr(sub[xcol], sub[ycol])
    return rho, p, n


if __name__ == '__main__':
    data = {ds: load_dataset(ds) for ds in DATASETS}

    print('=== MATCHED VARIANT COUNTS ===')
    for ds in DATASETS:
        n = len(data[ds])
        flag = '  <<< UNDERPOWERED (n<5... actually check per-analysis)' if n < 5 else ''
        print(f'  {DS_LABEL[ds]}: n={n}{flag}')
    print()

    # ── MAIN RESULT TABLES: one per proxy, 3 levels x 5 datasets ────────
    all_rows = []
    for proxy in PROXIES:
        print(f'\n{"="*70}\nPROXY: {proxy}\n{"="*70}')
        print(f'{"dataset":<14}{"rho_full":>10}{"p_full":>10}{"n_full":>7}   '
              f'{"rho_no_e11b":>12}{"p_no_e11b":>11}{"n_no_e11b":>10}   '
              f'{"rho_vs_rawgnn":>14}{"p_vs_rawgnn":>13}{"n_rawgnn":>9}')
        for ds in DATASETS:
            df = data[ds]
            if proxy not in df.columns:
                continue
            no_e11b = df[df['edge_idx'] != 'E11b']

            rf, pf, nf = corr(df, proxy, 'mean_ragas_composite')
            r0, p0, n0 = corr(no_e11b, proxy, 'mean_ragas_composite')
            rg, pg, ng = corr(no_e11b, proxy, 'raw_gnn_train_mean')

            flag = ' [UNDERPOWERED]' if nf < 5 else ''
            print(f'{DS_LABEL[ds]:<14}{rf:>10.3f}{pf:>10.3f}{nf:>7d}   '
                  f'{r0:>12.3f}{p0:>11.3f}{n0:>10d}   '
                  f'{rg:>14.3f}{pg:>13.3f}{ng:>9d}{flag}')

            all_rows.append({
                'proxy': proxy, 'dataset': ds,
                'rho_full': rf, 'p_full': pf, 'n_full': nf,
                'rho_no_e11b': r0, 'p_no_e11b': p0, 'n_no_e11b': n0,
                'rho_vs_rawgnn': rg, 'p_vs_rawgnn': pg, 'n_rawgnn': ng,
                'underpowered': nf < 5,
            })

    # ── PRACTITIONER RANKING TEST (full set, proxy-rank vs ragas-rank) ──
    print(f'\n\n{"="*70}\nPRACTITIONER RANKING TEST (full set, all matched variants)\n{"="*70}')
    ranking_rows = []
    for proxy in PROXIES + ['raw_gnn_train_mean']:
        higher_better = HIGHER_IS_BETTER.get(proxy, True)
        per_ds = {}
        top1_hits = 0
        n_ds_valid = 0
        for ds in DATASETS:
            df = data[ds].dropna(subset=[proxy, 'mean_ragas_composite']).copy()
            n = len(df)
            if n < 3 or df[proxy].nunique() < 2:
                per_ds[ds] = (float('nan'), float('nan'), n)
                continue
            rho, p = spearmanr(df[proxy], df['mean_ragas_composite'])
            # sign-adjust so rho is "does proxy ranking match ragas ranking" regardless of direction convention
            per_ds[ds] = (rho, p, n)
            n_ds_valid += 1

            # top-1 precision: best-by-proxy variant's RAGAS in top tercile?
            df_sorted = df.sort_values(proxy, ascending=not higher_better)
            top1_variant = df_sorted.iloc[0]
            tercile_cut = df['mean_ragas_composite'].quantile(2/3)
            if top1_variant['mean_ragas_composite'] >= tercile_cut:
                top1_hits += 1

        rhos = [v[0] for v in per_ds.values() if not pd.isna(v[0])]
        mean_rho = float(np.mean(rhos)) if rhos else float('nan')
        row = {'proxy': proxy, 'mean_rho': mean_rho, 'top1_precision': f'{top1_hits}/{len(DATASETS)}'}
        for ds in DATASETS:
            row[f'{ds}_rho'] = per_ds[ds][0]
        ranking_rows.append(row)

    rank_df = pd.DataFrame(ranking_rows).sort_values('mean_rho', ascending=False)
    pd.set_option('display.width', 200)
    print(rank_df.to_string(index=False))

    # ── SAVE ──────────────────────────────────────────────────────────
    main_df = pd.DataFrame(all_rows)
    main_df.to_csv(REPO_ROOT / 'data/graphrag/proxy_full_test_all_datasets.csv', index=False)
    rank_df.to_csv(REPO_ROOT / 'data/graphrag/proxy_ranking_test_all_datasets.csv', index=False)
    print(f'\nsaved -> proxy_full_test_all_datasets.csv, proxy_ranking_test_all_datasets.csv')

    # ── SCATTER PLOTS ────────────────────────────────────────────────
    best_proxy = rank_df.iloc[0]['proxy']
    print(f'\nbest proxy by mean_rho: {best_proxy}')

    def make_scatter(xcol, out_name, xlabel):
        colors = {'arxiv': 'tab:blue', 'amazon': 'tab:orange', 'history': 'tab:green',
                  'electronics': 'tab:red', 'toys': 'tab:purple'}
        markers = {'N7': 'o', 'N8': 's', 'N9': 'D'}
        fig, ax = plt.subplots(figsize=(9, 7))
        for ds in DATASETS:
            df = data[ds]
            for _, row in df.iterrows():
                if pd.isna(row.get(xcol)) or pd.isna(row['mean_ragas_composite']):
                    continue
                ax.scatter(row[xcol], row['mean_ragas_composite'], c=colors[ds],
                           marker=markers.get(row['node_type'], 'x'), s=70,
                           edgecolor='black', linewidth=0.4, alpha=0.85)
        ds_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=DS_LABEL[ds])
                      for ds, c in colors.items()]
        nt_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=10, label=nt)
                      for nt, m in markers.items()]
        leg1 = ax.legend(handles=ds_handles, title='Dataset', loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.add_artist(leg1)
        ax.legend(handles=nt_handles, title='Node type', loc='lower left', bbox_to_anchor=(1.01, 0))
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('mean RAGAS composite', fontsize=10)
        plt.tight_layout()
        fig.savefig(REPO_ROOT / f'data/graphrag/{out_name}', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'saved -> {out_name}')

    make_scatter(best_proxy, f'proxy_scatter_best_{best_proxy}_all_datasets.png', best_proxy)
    make_scatter('raw_gnn_train_mean', 'proxy_scatter_rawgnn_all_datasets.png', 'raw_gnn train_mean')
