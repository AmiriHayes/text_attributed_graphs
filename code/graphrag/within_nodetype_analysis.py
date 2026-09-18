#!/usr/bin/env python3
"""Within-node-type rawgnn/edge/text vs RAGAS analysis. Pure CSV joins,
zero new computation -- reuses ragas_results_{ds}.csv and the
{ds}_m1_rawgnn_lookup.csv files already built this session."""
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EDGE_RANK = {'E10a': 0, 'E10b': 1, 'E10c': 2, 'E11a': 3, 'E11b': 4, 'E11c': 5}
TEXT_RANK = {'T12a': 2, 'T12b': 1, 'T12e': 0}

DATASET_FILES = {
    'arxiv': 'ragas_results_arxiv_all_variants.csv',
    'amazon': 'ragas_results_amazon.csv',
    'electronics': 'ragas_results_electronics.csv',
    'toys': 'ragas_results_toys.csv',
    'history': 'ragas_results_history.csv',
}


def load_merged(ds: str) -> pd.DataFrame:
    ragas = pd.read_csv(REPO_ROOT / f'data/graphrag/{DATASET_FILES[ds]}')
    target = ragas.groupby('system_name').agg(
        mean_ragas_composite=('composite_score', 'mean'),
        node_idx=('node_idx', 'first'), edge_idx=('edge_idx', 'first'), text_idx=('text_idx', 'first'),
    ).reset_index()
    target['variant'] = target['node_idx'] + '_' + target['edge_idx'] + '_' + target['text_idx']

    rawgnn = pd.read_csv(REPO_ROOT / f'data/graphrag/{ds}_m1_rawgnn_lookup.csv')
    merged = target.merge(rawgnn, on='variant', how='left')
    return merged


def corr(df, xcol, ycol):
    sub = df[[xcol, ycol]].dropna()
    n = len(sub)
    if n < 3 or sub[xcol].nunique() < 2:
        return float('nan'), float('nan'), n
    rho, p = spearmanr(sub[xcol], sub[ycol])
    return rho, p, n


if __name__ == '__main__':
    summary_rows = []
    all_step34 = []

    for ds in ['amazon', 'electronics', 'arxiv', 'toys', 'history']:
        df = load_merged(ds)
        df['edge_rank'] = df['edge_idx'].map(EDGE_RANK)
        df['text_rank'] = df['text_idx'].map(TEXT_RANK)
        node_types = sorted(df['node_idx'].unique())

        print(f'\n{"="*70}\n{ds.upper()}\n{"="*70}')
        for nt in node_types:
            sub = df[df['node_idx'] == nt]
            n = len(sub)
            underpowered = n < 5
            print(f'\n--- {nt} (n={n}){"  [UNDERPOWERED]" if underpowered else ""} ---')

            # Step 2: rawgnn vs RAGAS
            r2, p2, n2 = corr(sub, 'raw_gnn_train_mean', 'mean_ragas_composite')
            print(f'  Step 2 rho(rawgnn, RAGAS)  = {r2:+.3f}  p={p2:.4f}  n={n2}')
            summary_rows.append({'dataset': ds, 'node_type': nt, 'n': n2,
                                  'rho_rawgnn_ragas': r2, 'p_rawgnn_ragas': p2,
                                  'underpowered': n2 < 5})

            # Step 3: edge rank vs RAGAS
            r3, p3, n3 = corr(sub, 'edge_rank', 'mean_ragas_composite')
            print(f'  Step 3 rho(edge_rank, RAGAS) = {r3:+.3f}  p={p3:.4f}  n={n3}')

            # Step 4: within each edge type, text_rank vs RAGAS
            print(f'  Step 4 (text fidelity within fixed edge type):')
            for et in sorted(sub['edge_idx'].unique()):
                et_sub = sub[sub['edge_idx'] == et]
                if et_sub['text_rank'].nunique() < 2 or len(et_sub) < 2:
                    print(f'    {et}: n={len(et_sub)} insufficient text-fidelity variation')
                    continue
                r4, p4, n4 = corr(et_sub, 'text_rank', 'mean_ragas_composite')
                flag = '  [UNDERPOWERED]' if n4 < 5 else ''
                print(f'    {et}: rho={r4:+.3f}  p={p4:.4f}  n={n4}{flag}')
                all_step34.append({'dataset': ds, 'node_type': nt, 'edge_type': et,
                                    'rho_text_ragas': r4, 'p_text_ragas': p4, 'n': n4})
            all_step34.append({'dataset': ds, 'node_type': nt, 'edge_type': '__ALL__',
                                'rho_edgerank_ragas': r3, 'p_edgerank_ragas': p3, 'n': n3})

        # ── scatter plot: one panel per node type, x=rawgnn, y=RAGAS, labeled by edge type ──
        fig, axes = plt.subplots(1, len(node_types), figsize=(5.5 * len(node_types), 4.8), facecolor='white')
        if len(node_types) == 1:
            axes = [axes]
        for ax, nt in zip(axes, node_types):
            sub = df[df['node_idx'] == nt]
            for et in sorted(sub['edge_idx'].unique()):
                et_sub = sub[sub['edge_idx'] == et]
                ax.scatter(et_sub['raw_gnn_train_mean'], et_sub['mean_ragas_composite'],
                           label=et, s=60, edgecolor='black', linewidth=0.4)
            ax.set_xlabel('raw_gnn train_mean', fontsize=9)
            ax.set_ylabel('mean RAGAS composite' if ax is axes[0] else '', fontsize=9)
            ax.set_title(f'{nt} (n={len(sub)})', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, frameon=False, title='Edge type', title_fontsize=7)
            ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        out = REPO_ROOT / f'data/graphrag/within_nodetype_scatter_{ds}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'\nsaved -> {out.name}')

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPO_ROOT / 'data/graphrag/within_nodetype_summary.csv', index=False)
    step34_df = pd.DataFrame(all_step34)
    step34_df.to_csv(REPO_ROOT / 'data/graphrag/within_nodetype_edge_text.csv', index=False)

    print(f'\n\n{"="*70}\nKEY SUMMARY TABLE\n{"="*70}')
    pd.set_option('display.width', 140)
    print(summary_df.to_string(index=False))
    print(f'\nsaved -> within_nodetype_summary.csv, within_nodetype_edge_text.csv')
