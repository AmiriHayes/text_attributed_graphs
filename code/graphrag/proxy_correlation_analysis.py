#!/usr/bin/env python3
"""Merge List 1 + List 2 proxy scores with RAGAS composite target,
compute Spearman correlations, build summary table + plots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

GRAPH_ONLY_COLS = [
    'largest_cc_fraction', 'mean_degree', 'isolated_node_fraction', 'diameter',
    'n_communities', 'modularity', 'mean_community_size', 'largest_community_fraction',
    'singleton_fraction', 'pagerank_mean', 'pagerank_std', 'pagerank_gini',
    'clustering_coeff_mean', 'betweenness_mean', 'homophily', 'link_pred_auc',
]
QUESTION_AWARE_COLS = [
    'precision_at_1', 'precision_at_3', 'precision_at_5', 'mean_hop_length',
    'hop_reachable_frac', 'hop_within3_frac', 'coherence_mean', 'coherence_std',
    'query_alignment_delta',
]

if __name__ == '__main__':
    ragas = pd.read_csv(REPO_ROOT / 'data/graphrag/ragas_results_arxiv_all_variants.csv')
    target = ragas.groupby('system_name')['composite_score'].mean().reset_index()
    target['variant'] = target['system_name'].str.replace('_text_based', '', regex=False)
    target = target[['variant', 'composite_score']].rename(columns={'composite_score': 'mean_ragas_composite'})
    print(f'target: {len(target)} variants with RAGAS scores')

    l1 = pd.read_csv(REPO_ROOT / 'data/graphrag/proxy_list1_arxiv.csv')
    l2 = pd.read_csv(REPO_ROOT / 'data/graphrag/proxy_list2_arxiv.csv')

    merged = target.merge(l1, on='variant', how='inner').merge(l2, on='variant', how='inner', suffixes=('', '_l2'))
    print(f'merged: {len(merged)} variants (n7={ (merged["node_type"]=="N7").sum() }, n8={(merged["node_type"]=="N8").sum()})')
    merged.to_csv(REPO_ROOT / 'data/graphrag/proxy_correlation_arxiv_merged.csv', index=False)

    results = []
    for col in GRAPH_ONLY_COLS + QUESTION_AWARE_COLS:
        if col not in merged.columns:
            continue
        sub = merged[['mean_ragas_composite', col]].dropna()
        n = len(sub)
        if n < 3:
            results.append({'proxy_score': col, 'spearman_rho': float('nan'), 'p_value': float('nan'),
                             'n': n, 'cost_category': 'graph_only' if col in GRAPH_ONLY_COLS else 'question_aware'})
            continue
        rho, p = spearmanr(sub[col], sub['mean_ragas_composite'])
        results.append({'proxy_score': col, 'spearman_rho': rho, 'p_value': p, 'n': n,
                         'cost_category': 'graph_only' if col in GRAPH_ONLY_COLS else 'question_aware'})

    res_df = pd.DataFrame(results)
    res_df['abs_rho'] = res_df['spearman_rho'].abs()
    res_df = res_df.sort_values('abs_rho', ascending=False).drop(columns='abs_rho')
    res_df.to_csv(REPO_ROOT / 'data/graphrag/proxy_correlation_arxiv.csv', index=False)

    pd.set_option('display.width', 140)
    print('\n=== PROXY CORRELATION SUMMARY (sorted by |rho|) ===')
    print(res_df.to_string(index=False))

    # ── Top-3 scatter ────────────────────────────────────────────────────
    top3 = res_df.dropna(subset=['spearman_rho']).head(3)['proxy_score'].tolist()
    fig, axes = plt.subplots(1, len(top3), figsize=(5.2 * len(top3), 4.5), facecolor='white')
    if len(top3) == 1:
        axes = [axes]
    for ax, col in zip(axes, top3):
        sub = merged.dropna(subset=[col, 'mean_ragas_composite'])
        colors = sub['node_type'].map({'N7': 'tab:blue', 'N8': 'tab:orange'})
        ax.scatter(sub[col], sub['mean_ragas_composite'], c=colors, s=60, edgecolor='black', linewidth=0.4)
        for _, r in sub.iterrows():
            label = r['variant']
            ax.annotate(label, (r[col], r['mean_ragas_composite']), fontsize=5.5, alpha=0.7,
                        xytext=(2, 2), textcoords='offset points')
        rho_row = res_df[res_df['proxy_score'] == col].iloc[0]
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel('mean RAGAS composite' if ax is axes[0] else '', fontsize=9)
        ax.set_title(f'{col}\nρ={rho_row["spearman_rho"]:+.3f} (n={int(rho_row["n"])})', fontsize=9, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=nt)
               for nt, c in [('N7', 'tab:blue'), ('N8', 'tab:orange')]]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout()
    fig.savefig(REPO_ROOT / 'data/graphrag/proxy_scatter_top3.png', dpi=200, bbox_inches='tight', facecolor='white')
    print('\nsaved -> proxy_scatter_top3.png')

    # ── Full correlation matrix (proxies vs each other + target) ───────
    all_cols = ['mean_ragas_composite'] + [c for c in GRAPH_ONLY_COLS + QUESTION_AWARE_COLS if c in merged.columns]
    corr_mat = merged[all_cols].corr(method='spearman')
    fig2, ax2 = plt.subplots(figsize=(12, 10), facecolor='white')
    im = ax2.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(all_cols)))
    ax2.set_yticks(range(len(all_cols)))
    ax2.set_xticklabels(all_cols, rotation=90, fontsize=7)
    ax2.set_yticklabels(all_cols, fontsize=7)
    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            v = corr_mat.values[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                          color='white' if abs(v) > 0.6 else 'black')
    fig2.colorbar(im, ax=ax2, shrink=0.7, label='Spearman ρ')
    plt.tight_layout()
    fig2.savefig(REPO_ROOT / 'data/graphrag/proxy_correlation_matrix.png', dpi=200, bbox_inches='tight', facecolor='white')
    print('saved -> proxy_correlation_matrix.png')
