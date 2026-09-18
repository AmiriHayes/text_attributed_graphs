#!/usr/bin/env python3
"""
Step 2-3 analysis for the full GraphRAG construction-ranking stability
study: stability curves (rho vs n questions), Scale B vs Scale C ranking
agreement, and the two-panel paper figure. Parameterized by dataset so
Amazon slots in with the same functions once its data completes -- no
code changes needed, just add 'amazon' to DATASETS and rerun main().

Note: T12e is excluded from all analysis here (see conversation) --
format_arxiv_row() (used by every GraphRAG script in this repo, including
run_stability_study_step1.py) always reads text_fidelity_a regardless of
which T variant is nominally being evaluated, so T12e never actually
represented a text-free condition for GraphRAG. T12a/T12b/T12e
composite-score differences are noise, not a text-fidelity effect.

Reads:
  - data/graphrag/ragas_stability_{dataset}_{scale}.csv (scale in B, C)

Writes:
  - data/graphrag/stability_curves_{dataset}.csv
  - data/graphrag/scale_comparison_{dataset}.csv
  - output/figures/stability_curve_{dataset}.pdf
  - output/figures/stability_gnn_vs_graphrag_{dataset}.pdf

Usage:
  python data/graphrag/analyze_stability_study.py
"""
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = ['arxiv', 'amazon']
SCALES = ['B', 'C']
N_VALUES = [3, 5, 10, 15, 20, 30, 40, 50, 75, 100]
SUBTYPES = ['single_specific', 'aggregate_cross_paper', 'edge_multihop', 'all_combined']
METRICS = ['faithfulness', 'answer_relevance', 'context_relevance', 'composite']
N_SEEDS = 20
RHO_THRESHOLD = 0.90

CHECK1B_PNG = REPO_ROOT / 'output/run_1000_final/analysis/checks/check1b_rho_vs_nsamples_all3.png'


def _exclude_t12e(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['variant'].str.endswith('T12e')].copy()


def load_stability_data(dataset: str, scale: str) -> pd.DataFrame:
    path = REPO_ROOT / f'data/graphrag/ragas_stability_{dataset}_{scale}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return _exclude_t12e(df)


# ── 1. Stability curve function ─────────────────────────────────────────────

def compute_stability_curve(dataset: str, scale: str,
                             n_values: list = None, subtypes: list = None,
                             metrics: list = None, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """rho vs n, for every (subtype, metric) combination, one dataset+scale."""
    n_values = n_values or N_VALUES
    subtypes = subtypes or SUBTYPES
    metrics = metrics or METRICS

    df = load_stability_data(dataset, scale)
    if df is None:
        return pd.DataFrame()

    variants = sorted(df['variant'].unique())
    rows = []
    for subtype in subtypes:
        sub_df = df if subtype == 'all_combined' else df[df['question_subtype'] == subtype]
        qids = sorted(sub_df['question_id'].unique())
        n_full = len(qids)
        if n_full < 3:
            continue

        for metric in metrics:
            pivot = sub_df.pivot_table(index='variant', columns='question_id', values=metric)
            pivot = pivot.reindex(index=variants, columns=qids)
            full_mean = pivot.mean(axis=1)

            ns = sorted(set([n for n in n_values if n <= n_full] + [n_full]))
            for n in ns:
                rhos = []
                for seed in range(n_seeds):
                    rng = np.random.RandomState(seed)
                    sub_q = qids if n == n_full else list(rng.choice(qids, size=n, replace=False))
                    sub_mean = pivot[sub_q].mean(axis=1)
                    if sub_mean.nunique() < 2 or full_mean.nunique() < 2:
                        continue
                    rho, _ = spearmanr(sub_mean, full_mean)
                    rhos.append(rho)
                if rhos:
                    rows.append({
                        'dataset': dataset, 'scale': scale, 'subtype': subtype, 'metric': metric,
                        'n': n, 'n_full': n_full, 'mean_rho': float(np.mean(rhos)),
                        'std_rho': float(np.std(rhos)), 'n_seeds': len(rhos),
                    })
    return pd.DataFrame(rows)


# ── 2. Scale comparison function ────────────────────────────────────────────

def compare_scales(dataset: str, metrics: list = None, subtypes: list = None) -> pd.DataFrame:
    """Spearman rho between Scale B and Scale C variant rankings, per metric,
    per subtype, and per node type (plus 'all' node types combined)."""
    metrics = metrics or METRICS
    subtypes = subtypes or SUBTYPES

    b = load_stability_data(dataset, 'B')
    c = load_stability_data(dataset, 'C')
    if b is None or c is None:
        return pd.DataFrame()

    rows = []
    node_types = ['all'] + sorted(set(b['node_type'].unique()) & set(c['node_type'].unique()))

    for subtype in subtypes:
        b_sub = b if subtype == 'all_combined' else b[b['question_subtype'] == subtype]
        c_sub = c if subtype == 'all_combined' else c[c['question_subtype'] == subtype]

        for metric in metrics:
            b_mean = b_sub.groupby(['variant', 'node_type'])[metric].mean().reset_index().rename(columns={metric: 'scaleB'})
            c_mean = c_sub.groupby(['variant', 'node_type'])[metric].mean().reset_index().rename(columns={metric: 'scaleC'})
            merged = b_mean.merge(c_mean, on=['variant', 'node_type'])

            for nt in node_types:
                sub = merged if nt == 'all' else merged[merged['node_type'] == nt]
                if len(sub) < 3 or sub['scaleB'].nunique() < 2:
                    rows.append({'dataset': dataset, 'subtype': subtype, 'metric': metric,
                                 'node_type': nt, 'rho': np.nan, 'p': np.nan, 'n': len(sub)})
                    continue
                rho, p = spearmanr(sub['scaleB'], sub['scaleC'])
                rows.append({'dataset': dataset, 'subtype': subtype, 'metric': metric,
                             'node_type': nt, 'rho': float(rho), 'p': float(p), 'n': len(sub)})
    return pd.DataFrame(rows)


# ── find best (scale, metric, subtype) combination ──────────────────────────

def find_best_combination(curve_df: pd.DataFrame, threshold: float = RHO_THRESHOLD) -> dict:
    """Smallest n at which mean_rho >= threshold, preferring combinations
    that cross it; among those that never cross, report the highest max rho."""
    crossing = []
    for (scale, subtype, metric), grp in curve_df.groupby(['scale', 'subtype', 'metric']):
        grp = grp[grp['n'] < grp['n_full']]  # exclude trivial self-comparison at n==n_full
        if grp.empty:
            continue
        above = grp[grp['mean_rho'] >= threshold]
        if len(above):
            n_star = above['n'].min()
            crossing.append({'scale': scale, 'subtype': subtype, 'metric': metric,
                             'n_star': int(n_star), 'rho_at_n_star': float(above[above['n']==n_star]['mean_rho'].iloc[0])})
    if crossing:
        best = sorted(crossing, key=lambda r: r['n_star'])[0]
        best['crossed_threshold'] = True
        return best

    # nothing crosses -- report the combination with the highest rho at its largest tested n
    best_row, best_val = None, -2
    for (scale, subtype, metric), grp in curve_df.groupby(['scale', 'subtype', 'metric']):
        grp = grp[grp['n'] < grp['n_full']]
        if grp.empty:
            continue
        row = grp.loc[grp['n'].idxmax()]
        if row['mean_rho'] > best_val:
            best_val = row['mean_rho']
            best_row = row
    if best_row is None:
        return {'crossed_threshold': False}
    return {'scale': best_row['scale'], 'subtype': best_row['subtype'], 'metric': best_row['metric'],
            'n_star': int(best_row['n']), 'rho_at_n_star': float(best_row['mean_rho']),
            'crossed_threshold': False}


# ── 3. Two-panel paper figure ────────────────────────────────────────────────

def plot_two_panel(dataset: str, curve_df: pd.DataFrame, best: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    if CHECK1B_PNG.exists():
        img = mpimg.imread(CHECK1B_PNG)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title(r'GNN: Strategy-1 $\rho$ vs. n subsets' + '\n(History, Amazon, ArXiv)',
                          fontsize=11, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'Check1B data not found', ha='center', va='center')
        axes[0].axis('off')

    ax = axes[1]
    if best.get('scale') and best.get('subtype') and best.get('metric'):
        sub = curve_df[(curve_df['scale'] == best['scale']) &
                       (curve_df['subtype'] == best['subtype']) &
                       (curve_df['metric'] == best['metric'])]
        sub = sub[sub['n'] <= sub['n_full']]
        ax.plot(sub['n'], sub['mean_rho'], marker='o', color='#4285F4', linewidth=2)
        ax.fill_between(sub['n'], sub['mean_rho'] - sub['std_rho'], sub['mean_rho'] + sub['std_rho'],
                        alpha=0.25, color='#4285F4')
        label = f"{dataset}, scale {best['scale']}, {best['subtype']}, {best['metric']}"
        ax.set_title(f'GraphRAG: best combination\n{label}', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    ax.axhline(RHO_THRESHOLD, color='#EA4335', linestyle='--', linewidth=1)
    ax.text(0.3, RHO_THRESHOLD + 0.01, r'$\rho=0.90$', color='#EA4335', fontsize=9)
    ax.set_xlabel('n questions')
    ax.set_ylabel(r'Spearman $\rho$ vs. full-set ranking')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    fig.suptitle('Construction-ranking stability: GNN subsets vs. GraphRAG questions',
                fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'saved -> {out_path}')


def plot_stability_curves(dataset: str, curve_df: pd.DataFrame, out_path: Path):
    """All (scale, subtype) curves for the composite metric, small multiples."""
    scales_present = sorted(curve_df['scale'].unique())
    fig, axes = plt.subplots(1, len(scales_present), figsize=(6.5 * len(scales_present), 5), sharey=True)
    if len(scales_present) == 1:
        axes = [axes]
    colors = {'single_specific': '#4285F4', 'aggregate_cross_paper': '#FBBC05',
             'edge_multihop': '#34A853', 'all_combined': '#999999'}

    for ax, scale in zip(axes, scales_present):
        sub_scale = curve_df[(curve_df['scale'] == scale) & (curve_df['metric'] == 'composite')]
        for subtype in SUBTYPES:
            sub = sub_scale[(sub_scale['subtype'] == subtype) & (sub_scale['n'] < sub_scale['n_full'])]
            if sub.empty:
                continue
            ax.plot(sub['n'], sub['mean_rho'], marker='o', color=colors.get(subtype, '#333'),
                    linewidth=2, label=subtype)
        ax.axhline(RHO_THRESHOLD, color='#EA4335', linestyle='--', linewidth=1)
        ax.set_title(f'{dataset} Scale {scale} (composite metric)', fontsize=11, fontweight='bold')
        ax.set_xlabel('n questions')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r'Spearman $\rho$ vs. full-set ranking')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'saved -> {out_path}')


def main(datasets: list = None):
    datasets = datasets or DATASETS
    all_curves = []

    for dataset in datasets:
        print(f'\n{"="*70}\n{dataset.upper()}\n{"="*70}')
        for scale in SCALES:
            df = load_stability_data(dataset, scale)
            if df is None:
                print(f'  Scale {scale}: no data yet, skipping')
                continue
            n_variants = df['variant'].nunique()
            print(f'  Scale {scale}: {n_variants} variants (T12e excluded), {len(df)} rows')
            curve = compute_stability_curve(dataset, scale)
            all_curves.append(curve)

        curve_df = pd.concat([c for c in all_curves if not c.empty and c['dataset'].iloc[0] == dataset],
                             ignore_index=True) if all_curves else pd.DataFrame()
        if curve_df.empty:
            print(f'  No stability curve data for {dataset} yet.')
            continue

        out_csv = REPO_ROOT / f'data/graphrag/stability_curves_{dataset}.csv'
        curve_df.to_csv(out_csv, index=False)
        print(f'  saved -> {out_csv}')

        # Scale comparison
        scale_cmp = compare_scales(dataset)
        if not scale_cmp.empty:
            cmp_csv = REPO_ROOT / f'data/graphrag/scale_comparison_{dataset}.csv'
            scale_cmp.to_csv(cmp_csv, index=False)
            print(f'  saved -> {cmp_csv}')

            print(f'\n  --- Scale B vs Scale C ranking agreement (composite, all_combined) ---')
            overall = scale_cmp[(scale_cmp['metric']=='composite') & (scale_cmp['subtype']=='all_combined') & (scale_cmp['node_type']=='all')]
            if len(overall):
                r = overall.iloc[0]
                print(f'  Overall: rho={r["rho"]:+.3f} p={r["p"]:.4f} n={r["n"]}')
            print(f'\n  --- Broken down by node type ---')
            by_nt = scale_cmp[(scale_cmp['metric']=='composite') & (scale_cmp['subtype']=='all_combined') & (scale_cmp['node_type']!='all')]
            for _, r in by_nt.iterrows():
                print(f'  {r["node_type"]}: rho={r["rho"]:+.3f} p={r["p"]:.4f} n={r["n"]}')
            print(f'\n  --- Broken down by question subtype (composite, all node types) ---')
            by_st = scale_cmp[(scale_cmp['metric']=='composite') & (scale_cmp['node_type']=='all')]
            for _, r in by_st.iterrows():
                print(f'  {r["subtype"]}: rho={r["rho"]:+.3f} p={r["p"]:.4f} n={r["n"]}')

        # Best combination
        best = find_best_combination(curve_df)
        print(f'\n  --- Best (scale, subtype, metric) combination ---')
        print(f'  {best}')

        # Figures
        fig_dir = REPO_ROOT / 'output/figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_stability_curves(dataset, curve_df, fig_dir / f'stability_curve_{dataset}.pdf')
        plot_two_panel(dataset, curve_df, best, fig_dir / f'stability_gnn_vs_graphrag_{dataset}.pdf')

    return all_curves


if __name__ == '__main__':
    main()
