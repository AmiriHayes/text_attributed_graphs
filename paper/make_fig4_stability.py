#!/usr/bin/env python3
"""
Construction-ranking stability versus evaluation budget, dual panel.

Left : node classification, rho vs number of train-pool subsets
Right: GraphRAG retrieval,  rho vs number of train-pool questions

Both panels run the SAME estimator as dt_consistency.py, on different
replication axes.  For a budget of n units:

  1. Draw n units at random from the train pool (subsets for node
     classification, questions for GraphRAG).
  2. Average each variant's score over just those n units.
  3. Fit a tree on one-hot (Node_Idx, Edge_Idx, Text_Idx) with those means,
     using dt_consistency.TREE_KWARGS unchanged.
  4. Spearman rho against per-variant means over the FULL held-out test pool,
     which is never subsampled.

Repeated over N_SEEDS draws per point; bands are +/- 1 standard deviation.

The variant set is fixed once from the full train pool, so the same variants
are ranked at every budget.  At the largest budget the draw is the whole
train pool, the standard deviation collapses to zero, and the point equals
the corresponding diagonal entry of Table 3 exactly.  That identity is the
figure's correctness check and is asserted at the end of the run.

Reads:   the same files as dt_consistency.py
Writes:  paper/artifacts/fig4_stability_dual.{pdf,png}
         output/run_final/analysis/dt_consistency/stability_curve.csv

Usage:   python3 paper/make_fig4_stability.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import dt_consistency as D

FIG_DIR = Path(__file__).resolve().parent / 'artifacts'
OUT = D.RUN / 'analysis' / 'dt_consistency'

N_SEEDS = 20
GNN_BUDGETS = [5, 10, 15, 20, 30, 40, 50, 60, 75]
RAG_BUDGETS = [5, 10, 20, 30, 50, 75, 100, 125, 150]
COLOR = {'history': '#8c564b', 'amazon': '#1f77b4', 'arxiv': '#d62728',
         'electronics': '#2ca02c', 'toys': '#ff7f0e'}


# ── per-unit score tables ─────────────────────────────────────────────────────

def gnn_units(ds):
    """(variant axes, unit, score) for node classification; unit = subset."""
    df = pd.read_csv(D.RUN / f'construction_performance_table_{ds}.csv')
    df = df[df['Task_Idx'] == D.NODE_CLASSIFICATION_ROWS].copy()
    if not D.INCLUDE_CONTROL:
        df = df[df['Text_Idx'] != D.NO_TEXT_CONTROL]
    col = 'S_GNN_step1' if 'S_GNN_step1' in df.columns else 'normalized_score'
    df = df.rename(columns={col: 'score', 'sample_idx': 'unit', 'run_split': 'pool'})
    return df[D.AXES + ['pool', 'unit', 'score']]


def rag_units(ds):
    """(variant axes, unit, score) for retrieval; unit = question."""
    df = pd.read_csv(D.RUN / f'ragas_results_{ds}.csv')
    axes = df['variant'].str.extract(r'^(N\d+)_(E\d+[a-z]?)_(T\d+[a-z]?)$')
    axes.columns = D.AXES
    df = pd.concat([df.drop(columns=[c for c in D.AXES if c in df.columns]), axes], axis=1)
    df = df[df['Text_Idx'] != D.NO_TEXT_CONTROL]
    split = pd.read_csv(D.RUN / f'question_split_{ds}.csv').set_index('question_id')['split']
    df = df[df['question_id'].isin(split.index)].copy()
    df['pool'] = df['question_id'].map(split)
    return df.rename(columns={'question_id': 'unit', 'composite': 'score'})[
        D.AXES + ['pool', 'unit', 'score']]


LOADERS = {'node_classification': (gnn_units, GNN_BUDGETS),
           'graphrag': (rag_units, RAG_BUDGETS)}


# ── the curve ─────────────────────────────────────────────────────────────────

def curve(ds, unit_loader, budgets, canonical_variants):
    u = unit_loader(ds)
    u = u.merge(canonical_variants, on=D.AXES, how='inner')

    train = u[u.pool == 'train']
    test_mean = (u[u.pool == 'test'].groupby(D.AXES)['score'].mean() * 100)

    units = np.array(sorted(train.unit.unique()))
    wide = train.pivot_table(index=D.AXES, columns='unit', values='score')
    wide = wide.reindex(test_mean.index)
    actual = test_mean.to_numpy(float)

    rows = []
    for n in budgets:
        if n > len(units):
            continue
        vals = []
        for s in range(N_SEEDS):
            rng = np.random.RandomState(1000 + s)
            pick = units if n == len(units) else rng.choice(units, n, replace=False)
            tm = wide[pick].mean(axis=1).to_numpy(float) * 100
            frame = wide.index.to_frame(index=False)
            frame['train_mean'] = tm
            pred = D.apply_to(D.fit(frame, 'train_mean'), frame)
            vals.append(spearmanr(pred, actual)[0])
        rows.append({'dataset': ds, 'n': n, 'mean': np.mean(vals),
                     'std': np.std(vals), 'n_units_total': len(units)})
    return pd.DataFrame(rows)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--include_control', action='store_true',
                    help='Match the cost table: keep the no-text control in the '
                         'node-classification variant set.')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    D.INCLUDE_CONTROL = args.include_control
    OUT = Path(args.out)
    print(f'no-text control: {"INCLUDED" if D.INCLUDE_CONTROL else "excluded"}')

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    titles = {'node_classification': 'Node classification',
              'graphrag': 'GraphRAG retrieval'}
    xlabels = {'node_classification': 'Train-pool subsets evaluated',
               'graphrag': 'Train-pool questions evaluated'}

    for ax, (task, (loader, budgets)) in zip(axes, LOADERS.items()):
        # Fix the variant set once, from the full train pool, so every budget
        # ranks the same constructions.
        canon = {ds: D.TASKS[task](ds)[D.AXES] for ds in D.DATASETS}
        for ds in D.DATASETS:
            c = curve(ds, loader, budgets, canon[ds])
            c['task'] = task
            all_rows.append(c)
            ax.plot(c['n'], c['mean'], marker='o', ms=3.5, lw=1.8,
                    color=COLOR[ds], label=D.LABEL[ds])
            ax.fill_between(c['n'], c['mean'] - c['std'], c['mean'] + c['std'],
                            color=COLOR[ds], alpha=0.13, lw=0)
        ax.axhline(0.80, color='0.45', ls=':', lw=1.1, zorder=0)
        ax.set_title(titles[task], fontsize=13, pad=12)
        ax.set_xlabel(xlabels[task], fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r'Spearman $\rho$ on test data', fontsize=11)
    axes[0].set_ylim(-0.05, 1.02)
    axes[1].legend(fontsize=8.5, loc='lower right', frameon=True, framealpha=0.92)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'fig4_stability_dual.{ext}', dpi=200,
                    bbox_inches='tight', facecolor='white')

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT / 'stability_curve.csv', index=False)

    # Correctness check: the full-budget point must equal the Table 3 diagonal.
    print(f'\n{"task":20s}{"dataset":13s}{"curve endpoint":>15}{"table 3 rho":>13}{"std":>7}')
    ok = True
    for task in LOADERS:
        summ = pd.read_csv(OUT / f'{task}_summary.csv').set_index('dataset')
        for ds in D.DATASETS:
            sub = df[(df.task == task) & (df.dataset == ds)]
            end = sub.iloc[-1]
            ref = summ.loc[ds, 'rho']
            match = abs(end['mean'] - ref) < 1e-9 and end['std'] < 1e-12
            ok &= match
            print(f'{task:20s}{ds:13s}{end["mean"]:15.4f}{ref:13.4f}{end["std"]:7.1e}'
                  + ('' if match else '   MISMATCH'))
    print(f'\nendpoints reproduce Table 3: {ok}')
    print(f'wrote {FIG_DIR}/fig4_stability_dual.pdf')


if __name__ == '__main__':
    main()
