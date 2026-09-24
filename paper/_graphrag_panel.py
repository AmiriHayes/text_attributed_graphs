#!/usr/bin/env python3
"""
Two-panel GraphRAG construction-performance figure -- the retrieval analogue
of _construction_panel.py, sharing its styling and label scheme.

The GNN figure's two axes have no literal GraphRAG counterpart, so each is
replaced by its closest analogue:

  LEFT  epochs -> questions evaluated. A GraphRAG variant has no training
        loop; what accumulates instead is evidence. The curve is the running
        mean score as test questions are consumed one at a time. Thin lines
        are independent random question orderings (the analogue of individual
        subsets); the thick line is their mean. A curve that flattens early
        means few questions suffice to rank that construction.

  RIGHT test-pool subsets -> question buckets. The 150 held-out questions are
        split into N_BUCKETS contiguous groups after a fixed shuffle, and each
        cell is a variant's mean score over one bucket -- structurally the same
        as the GNN heatmap's variant x subset grid.

Metric is the RAGAS composite (mean of faithfulness, answer relevance and
context relevance), the project's pre-registered target.

T6e is excluded entirely: format_rows.py always emits text_fidelity_a
regardless of T variant, so T6e is not a genuine no-text control for
retrieval the way it is for the GNN. There is therefore no control curve on
the left panel -- a difference from the GNN figure, and a deliberate one.

Usage:
  python3 code/build_fig_graphrag.py --dataset amazon
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

REPO = Path(__file__).resolve().parent.parent
DISP = {'arxiv': 'ArXiv', 'amazon': 'Amazon Sports', 'history': 'History',
        'electronics': 'Electronics', 'toys': 'Toys'}
# All solid: each variant already has its own colour, and dashed/dash-dot
# styles broke up the thin per-sample lines into something hard to follow.
NODE_STYLE = {'N7': '-', 'N8': '-', 'N9': '-'}
PALETTE = {'N7': ['#EA4335', '#FF6D00'], 'N8': ['#4285F4', '#A142F4'],
           'N9': ['#FBBC05', '#34A853']}
METRIC = 'composite'
N_BUCKETS = 10
# 3, matching the GNN panels' sample count. Note these are NOT independent
# replicates the way GNN subsets are: every ordering is the same 150 questions
# in a different sequence, so all orderings of a variant converge to exactly
# the same value at the right edge. The spread between them shows how much the
# running estimate depends on WHICH questions have been seen so far, and it
# necessarily collapses to zero once all questions are consumed.
N_ORDERINGS = 3
SEED = 42


def relabel(t):
    """Old internal codes -> paper scheme (subtract 6, keep the letter suffix).

    Anchors on separator-or-boundary rather than \b: underscores are word
    characters, so \b never fires inside "N7_E11a_T12a" and the whole string
    reads as one word. This accepts both underscore- and slash-separated
    variant strings.
    """
    return re.sub(r'(?:(?<=^)|(?<=[_/\s]))([NET])(\d+)([a-z]?)(?=[_/\s]|$)',
                  lambda m: f'{m.group(1)}{int(m.group(2)) - 6}{m.group(3)}', t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DISP))
    args = ap.parse_args()
    ds = args.dataset

    d = pd.read_csv(REPO / f'output/run_final/ragas_results_{ds}.csv')
    p = d['variant'].str.extract(r'^(N\d+)_(E\d+[a-z]?)_(T\d+[a-z]?)$')
    d[['N', 'E', 'T']] = p
    d = d[d['T'] != 'T12e'].copy()
    test = d[d.split == 'test'].copy()

    # BEST and WORST variant per node type -- not the top two. Taking the top
    # two selects variants that are near-identical by construction (on ArXiv
    # all four shared edge type E4b, and their pairwise differences were not
    # significant, p>0.05 uncorrected). The discriminating axis for GraphRAG is
    # the edge type, not the node type, so the panel has to span each node
    # type's range to show anything real.
    # Selection comes from build_fig_construction.rule_a so that this figure
    # and the GNN figure are guaranteed to plot the same variants in the same
    # colours. Ranking is on all questions, not the test split alone: the two
    # bases disagree on three of five datasets, and one definition has to win.
    from _construction_panel import rule_a, surviving
    _, keep = surviving(ds)
    chosen = [('_'.join(pk[:3]), pk[0], pk[3]) for pk in rule_a(ds, keep)]
    order = test.groupby(['N', 'variant'])[METRIC].mean().reset_index()

    print(f'{ds}: {d.variant.nunique()} variants, node types '
          f'{[relabel(n) for n in sorted(order.N.unique())]}, '
          f'{test.question_id.nunique()} test questions')
    print(f"  curves: {[relabel(v.replace('_', '/')) for v, _, _ in chosen]}")

    fig = plt.figure(figsize=(15, 6.5), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.20], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])

    qids = sorted(test.question_id.unique())
    for v, n, color in chosen:
        s = test[test.variant == v].set_index('question_id')[METRIC]
        runs = []
        for k in range(N_ORDERINGS):
            rng = np.random.RandomState(SEED + k)
            vals = s.loc[list(rng.permutation(qids))].values * 100
            run = np.cumsum(vals) / np.arange(1, len(vals) + 1)
            runs.append(run)
            ax1.plot(range(1, len(run) + 1), run, color=color, lw=1.4, alpha=0.30, zorder=2)
        ax1.plot(range(1, len(runs[0]) + 1), np.mean(runs, axis=0), color=color,
                 linestyle=NODE_STYLE.get(n, '-'), lw=5, label=relabel(v.replace('_', '/')), zorder=3)

    ax1.text(0.5, 1.05, 'Individual Orderings = Thin Lines     Mean of Orderings = Thick Line',
             transform=ax1.transAxes, ha='center', fontsize=11.5, style='italic')
    ax1.set_xlabel('Questions evaluated', fontsize=15)
    ax1.set_ylabel('RAGAS Composite', fontsize=15)
    ax1.set_xlim(1, len(qids))
    # Fixed 0-100 so every panel in the paper, GNN and retrieval alike, is read
    # on the same scale. Retrieval composites occupy a narrow band near the top,
    # so curves sit closer together than a zoomed range would show.
    ax1.set_ylim(0, 100)
    ax1.set_yticks(range(0, 101, 10))
    # PercentFormatter rather than set_yticklabels(get_yticks()): the latter
    # pins labels to whatever ticks happen to exist at call time and warns
    # that they may be mislabeled if the locator later moves them.
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    leg = ax1.legend(fontsize=9.5, loc='upper right', frameon=True, framealpha=0.92,
                     handlelength=2.6, handletextpad=0.8, labelspacing=0.5,
                     borderpad=0.6, alignment='left')
    for h in leg.legend_handles:
        h.set_linestyle('-'); h.set_linewidth(4.0); h.set_alpha(1.0)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(alpha=0.25)
    ax1.tick_params(labelsize=12)

    rng = np.random.RandomState(SEED)
    shuffled = list(rng.permutation(qids))
    bucket = {q: i for i, chunk in enumerate(np.array_split(shuffled, N_BUCKETS)) for q in chunk}
    test['bucket'] = test.question_id.map(bucket)
    piv = test.pivot_table(index=['N', 'E', 'T'], columns='bucket', values=METRIC,
                           aggfunc='mean') * 100
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    piv = piv.sort_index(level=0, sort_remaining=False)
    piv.index = [relabel('/'.join(i)) for i in piv.index]

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(piv, ax=ax2, cmap='RdYlGn', linewidths=0.3, linecolor='#e0e0e0',
                cbar=False, yticklabels=True, xticklabels=False,
                annot=True, fmt='.0f', annot_kws={'fontsize': 6.5})
    ax2.set_xlabel(''); ax2.set_ylabel('')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8.5)

    # No figure title: the manuscript supplies the dataset and task in the
    # minipage header above each panel, so a suptitle would duplicate it.
    plt.tight_layout()
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0 - 0.025, pos.width, pos.height + 0.060])

    out = Path(__file__).resolve().parent / 'artifacts' / f'fig_{ds}_graphrag_performance_runfinal'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  heatmap rows: {len(piv)} x {N_BUCKETS} buckets  saved -> {out.name}.pdf')


if __name__ == '__main__':
    main()
