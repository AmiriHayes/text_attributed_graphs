#!/usr/bin/env python3
"""
Combined 2x2 construction-performance figure: GNN row on top, GraphRAG row
below. Each row is the left/right pair those two standalone builders produce,
so the result is 2 rows x 2 columns.

Narrower than the standalone figures (12 x 9.6 in rather than 15 x 6.5) so it
scales sensibly to \\columnwidth: a 15-in-wide figure shrunk to a ~6.5-in
column renders its heatmap annotations illegibly small.

Row 1  GNN:      epoch accuracy curves | Top1 heatmap (variant x test subset)
Row 2  GraphRAG: running-mean composite | composite heatmap (variant x question bucket)

T6e appears as a no-text control on the GNN curves only. It is excluded from
the GNN heatmap (a dozen near-identical floor rows would compress the colour
scale) and from GraphRAG entirely, since format_rows.py always emits
text_fidelity_a regardless of T variant and so T6e is not a genuine no-text
control for retrieval.

Usage:
  python3 code/build_fig_combined.py --dataset amazon \
      --epoch_dir output/amazon_epoch_t6e/epoch_logs
"""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
from generate_ablation_heatmaps import build_pivot, compute_row_order, HELD_OUT_SAMPLES, HELD_OUT_FALLBACK_N

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

REPO = Path(__file__).resolve().parent.parent
F = ['Node_Idx', 'Edge_Idx', 'Text_Idx']
ZERO = 0.95
# Panel titles use the full dataset name -- "Amazon" alone is ambiguous once
# Electronics and Toys (also Amazon Reviews categories) appear in the same paper.
DISP = {'arxiv': 'ArXiv', 'amazon': 'Amazon Sports', 'history': 'History',
        'electronics': 'Electronics', 'toys': 'Toys'}
# All solid: each variant already has its own colour, and dashed/dash-dot
# styles broke up the thin per-sample lines into something hard to follow.
NODE_STYLE = {'N7': '-', 'N8': '-', 'N9': '-'}
PALETTE = {'N7': ['#EA4335', '#FF6D00'], 'N8': ['#4285F4', '#A142F4'],
           'N9': ['#FBBC05', '#34A853']}
CONTROL = '#5F6368'
METRIC = 'composite'
N_BUCKETS, N_ORDERINGS, SEED = 10, 3, 42


def relabel(t):
    return re.sub(r'(?:(?<=^)|(?<=[_/\s]))([NET])(\d+)([a-z]?)(?=[_/\s]|$)',
                  lambda m: f'{m.group(1)}{int(m.group(2)) - 6}{m.group(3)}', t)


def style_legend(ax, **kw):
    leg = ax.legend(fontsize=8, frameon=True, framealpha=0.92, handlelength=2.4,
                    handletextpad=0.7, labelspacing=0.4, borderpad=0.5,
                    alignment='left', **kw)
    for h in leg.legend_handles:
        h.set_linestyle('-'); h.set_linewidth(3.5); h.set_alpha(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='amazon', choices=list(DISP))
    ap.add_argument('--epoch_dir', default=None)
    ap.add_argument('--width', type=float, default=12.0)
    args = ap.parse_args()
    ds, name = args.dataset, DISP[args.dataset]
    if args.epoch_dir:
        epoch_dir = Path(args.epoch_dir)
    else:
        # Rule A selects variants the main run did not log; prefer the side-run.
        side = REPO / f'output/run_final/epoch_logs_figures/{ds}'
        epoch_dir = side if side.is_dir() else REPO / f'output/run_final/epoch_logs/{ds}'

    fig = plt.figure(figsize=(args.width, args.width * 0.80), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.20], height_ratios=[1, 1],
                           wspace=0.26, hspace=0.39)

    # ───────────────────────── ROW 1: GNN ─────────────────────────
    d = pd.read_csv(REPO / f'output/run_final/construction_performance_table_{ds}.csv')
    m1 = d[d.Task_Idx == 'M1'].copy()
    keep = []
    for k, g in m1.groupby(F):
        if k[2] == 'T12e':
            continue
        tr = g[g.run_split == 'train']; v = tr['S_GNN_step1'].dropna()
        if len(tr) > 0 and int((v == 0).sum()) / len(tr) > ZERO:
            continue
        if v.isna().all():
            continue
        keep.append(k)
    keep = set(keep)
    stems = sorted({Path(p).name.rsplit('_', 1)[0] for p in glob.glob(str(epoch_dir / '*.csv'))})
    ctrl = next((tuple(s.split('_')[1:]) for s in stems if s.endswith('T12e')), None)

    # ── ONE shared variant selection, used by both rows ──
    # Best and worst per node type ranked by GraphRAG composite ("Rule A").
    # Measured against the alternatives on all five datasets, this recovers
    # 90-99% of the GNN score range AND 100% of the GraphRAG range; selecting on
    # GNN instead recovers the GNN range but as little as 12% of GraphRAG's.
    # The asymmetry is structural: GNN variance sits almost entirely on the node
    # axis, so any selection spanning node types spans it, whereas GraphRAG
    # variance is spread across node and edge and needs a GraphRAG-aware rule.
    # Using one list for both rows is what keeps a colour meaning the same
    # variant top and bottom.
    r = pd.read_csv(REPO / f'output/run_final/ragas_results_{ds}.csv')
    pp = r['variant'].str.extract(r'^(N\d+)_(E\d+[a-z]?)_(T\d+[a-z]?)$')
    r[['N', 'E', 'T']] = pp
    r = r[r['T'] != 'T12e']
    test = r[r.split == 'test'].copy()
    gra_score = test.groupby('variant')[METRIC].mean()

    chosen = []
    for n in sorted({v.split('_')[0] for v in gra_score.index}):
        vs = gra_score[[v for v in gra_score.index if v.split('_')[0] == n]].sort_values()
        picks = [vs.index[-1]] if len(vs) == 1 else [vs.index[-1], vs.index[0]]
        for i, v in enumerate(picks):
            chosen.append((*v.split('_'), PALETTE.get(n, ['#777', '#999'])[i % 2]))
    missing = [('/'.join(c[:3])) for c in chosen
               if not glob.glob(str(epoch_dir / f'M1_{c[0]}_{c[1]}_{c[2]}_*.csv'))]
    if missing:
        print(f'  WARNING: no epoch logs for {[relabel(m) for m in missing]} '
              f'-- those curves will be absent from the GNN panel')

    ax = ax_r0l = fig.add_subplot(gs[0, 0])
    peak = 0.0
    for N, E, T, color in chosen + ([(*ctrl, CONTROL)] if ctrl else []):
        paths = sorted(glob.glob(str(epoch_dir / f'M1_{N}_{E}_{T}_*.csv')))
        curves, is_base = [], T == 'T12e'
        for i, p in enumerate(paths):
            col = 'test_top1' if 'test_top1' in pd.read_csv(p, nrows=1).columns else 'test_acc'
            s = pd.read_csv(p).set_index('epoch')[col] * 100
            curves.append(s); peak = max(peak, float(s.max()))
            ax.plot(s.index, s.values, color=color, lw=1.3, alpha=0.32, zorder=2,
                    label=relabel('/'.join((N, E, T))) if (is_base and i == 0) else None)
        if not curves or is_base:
            continue
        mc = pd.concat(curves, axis=1).mean(axis=1)
        ax.plot(mc.index, mc.values, color=color, linestyle=NODE_STYLE[N], lw=4,
                label=relabel('/'.join((N, E, T))), zorder=3)
    top = 100   # fixed scale, matching the appendix construction panels
    ax.set_xlabel('Epochs', fontsize=16, labelpad=6)
    ax.set_ylabel('GNN Score', fontsize=16)
    ax.set_ylim(0, top); ax.set_xlim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    style_legend(ax, loc='upper left'); ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=13)

    gdf = m1[m1[F].apply(tuple, axis=1).isin(keep)].copy()
    gdf['sample_idx'] = pd.to_numeric(gdf['sample_idx'], errors='coerce').astype('Int64')
    td = gdf[gdf.run_split == 'test']
    samples = [s for s in HELD_OUT_SAMPLES if s in td.sample_idx.values] or \
              sorted(td.sample_idx.unique())[-HELD_OUT_FALLBACK_N:]
    piv = build_pivot(gdf, 'Top1', 100.0, samples, compute_row_order(gdf, samples))
    piv.index = [relabel(str(i).replace('M1_', '').replace('_', '/')) for i in piv.index]
    ax = ax_r0r = fig.add_subplot(gs[0, 1])
    sns.heatmap(piv, ax=ax, cmap='RdYlGn', linewidths=0.3, linecolor='#e0e0e0', cbar=False,
                yticklabels=True, xticklabels=False, annot=True, fmt='.0f',
                annot_kws={'fontsize': 4.6})
    ax.set_xlabel('Test subsets (GNN score)', fontsize=16, labelpad=23); ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.6)

    # ─────────────────────── ROW 2: GraphRAG ───────────────────────
    # same six variants, same colours as the GNN row above
    gchosen = [('_'.join(c[:3]), c[0], c[3]) for c in chosen]

    ax = ax_r1l = fig.add_subplot(gs[1, 0])
    qids = sorted(test.question_id.unique())
    for v, n, color in gchosen:
        s = test[test.variant == v].set_index('question_id')[METRIC]
        runs = []
        for k in range(N_ORDERINGS):
            rng = np.random.RandomState(SEED + k)
            vals = s.loc[list(rng.permutation(qids))].values * 100
            run = np.cumsum(vals) / np.arange(1, len(vals) + 1)
            runs.append(run)
            ax.plot(range(1, len(run) + 1), run, color=color, lw=1.2, alpha=0.30, zorder=2)
        ax.plot(range(1, len(runs[0]) + 1), np.mean(runs, axis=0), color=color,
                linestyle=NODE_STYLE.get(n, '-'), lw=4, label=relabel(v.replace('_', '/')), zorder=3)
    ax.set_ylim(0, 100)   # fixed scale, matching every other panel
    ax.set_xlim(1, len(qids))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel('Questions evaluated', fontsize=16, labelpad=6)
    ax.set_ylabel('RAGAS Composite', fontsize=16)
    style_legend(ax, loc='lower left'); ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=13)

    rng = np.random.RandomState(SEED)
    buckets = {q: i for i, ch in enumerate(np.array_split(list(rng.permutation(qids)), N_BUCKETS)) for q in ch}
    test['bucket'] = test.question_id.map(buckets)
    gp = test.pivot_table(index=['N', 'E', 'T'], columns='bucket', values=METRIC, aggfunc='mean') * 100
    gp = gp.loc[gp.mean(axis=1).sort_values(ascending=False).index].sort_index(level=0, sort_remaining=False)
    gp.index = [relabel('/'.join(i)) for i in gp.index]
    ax = ax_r1r = fig.add_subplot(gs[1, 1])
    sns.heatmap(gp, ax=ax, cmap='RdYlGn', linewidths=0.3, linecolor='#e0e0e0', cbar=False,
                yticklabels=True, xticklabels=False, annot=True, fmt='.0f',
                annot_kws={'fontsize': 4.6})
    ax.set_xlabel('Question buckets (RAGAS composite)', fontsize=16, labelpad=23); ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.6)

    # One title per row rather than one per panel: the two panels in a row are
    # two views of the same task, so a spanning title says that and frees the
    # vertical space four separate titles were using.
    for (left, right), task in [((ax_r0l, ax_r0r), 'Node Classification'),
                                ((ax_r1l, ax_r1r), 'Question Answering')]:
        bl, br = left.get_position(), right.get_position()
        fig.text((bl.x0 + br.x1) / 2, max(bl.y1, br.y1) + 0.022,
                 f'{name}: Construction Performance on {task}',
                 ha='center', va='bottom', fontsize=19, fontweight='bold')

    out = Path(__file__).resolve().parent / 'artifacts' / f'fig_{ds}_combined_performance'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'{ds}: GNN {len(piv)} rows | GraphRAG {len(gp)} rows x {N_BUCKETS} buckets')
    print(f'  GNN curves: {[relabel("/".join(c[:3])) for c in chosen]}'
          + (f' + control {relabel("/".join(ctrl))}' if ctrl else ''))
    print(f'  GraphRAG curves: {[relabel(v.replace("_", "/")) for v, _, _ in gchosen]}')
    print(f'  saved -> {out.name}.pdf  ({args.width} x {args.width*0.80:.1f} in)')


if __name__ == '__main__':
    main()
