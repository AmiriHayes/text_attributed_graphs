#!/usr/bin/env python3
"""
Two-panel construction-performance figure for any dataset, from run_final.
Left: epoch accuracy curves (up to 2 variants per node type + one no-text
control). Right: Top1 heatmap over surviving M1 variants x test-pool subsets.

Labels use the paper scheme (N1/N2/N3, E4x/E5x, T6x); underlying data keeps
its original N7/E10x/T12x codes, so relabelling happens at render time only.

EPOCH COVERAGE IS PARTIAL. run_final enabled epoch logging for six variants
per dataset, so some node types present in the score data have no curves:
    arxiv        N1,N2 in data; N1,N2 logged   -> complete
    history      N1     in data; N1    logged   -> complete
    amazon       N1,N2,N3 in data; N1,N2 logged -> N3 missing
    electronics  N1,N2,N3 in data; N1,N2 logged -> N3 missing
    toys         N1,N2,N3 in data; N1,N3 logged -> N2 missing
Missing node types are reported at run time, never silently dropped. For
Amazon a side-run (output/amazon_epoch_t6e) supplies the gap; pass
--epoch_dir to point elsewhere.

T6e appears as a single control curve on the left only. It is NOT zero on
Top1 -- it sits at the majority-class floor -- but it is excluded from the
heatmap, where a dozen near-identical floor rows would compress the colour
scale across the constructions being compared.

Usage:
  python3 code/build_fig_construction.py --dataset arxiv
"""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
from generate_ablation_heatmaps import build_pivot, compute_row_order, HELD_OUT_SAMPLES, HELD_OUT_FALLBACK_N

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

REPO = Path(__file__).resolve().parent.parent
F = ['Node_Idx', 'Edge_Idx', 'Text_Idx']
ZERO = 0.95
DISP = {'arxiv': 'ArXiv', 'amazon': 'Amazon', 'history': 'History',
        'electronics': 'Electronics', 'toys': 'Toys'}
NODE_STYLE = {'N7': '-', 'N8': '--', 'N9': '-.'}
PALETTE = {'N7': ['#EA4335', '#FF6D00'], 'N8': ['#4285F4', '#A142F4'],
           'N9': ['#FBBC05', '#34A853']}
CONTROL_COLOR = '#5F6368'


def relabel(t):
    return re.sub(r'(?:(?<=^)|(?<=[_/\s]))([NET])(\d+)([a-z]?)(?=[_/\s]|$)',
                  lambda m: f'{m.group(1)}{int(m.group(2)) - 6}{m.group(3)}', t)


def rule_a(ds, keep):
    """Rule A: best and worst surviving variant per node type, ranked by the
    GraphRAG composite.  Selecting on GraphRAG recovers 90-99% of the GNN
    score range and 100% of GraphRAG's, while selecting on the GNN recovers
    only 12-41% of GraphRAG's on three datasets -- so one selection drives
    both panels and a colour names the same construction in each."""
    r = pd.read_csv(REPO / f'output/run_final/ragas_results_{ds}.csv')
    axes = r['variant'].str.extract(r'^(N\d+)_(E\d+[a-z]?)_(T\d+[a-z]?)$')
    axes.columns = F
    r = pd.concat([r, axes], axis=1)
    r = r[r[F].apply(tuple, axis=1).isin(keep)]
    sc = r.groupby(F)['composite'].mean().sort_values()

    picks = []
    for n in sorted({k[0] for k in sc.index}):
        vs = sc[[k for k in sc.index if k[0] == n]]
        sel = [vs.index[-1]] if len(vs) == 1 else [vs.index[-1], vs.index[0]]
        for i, v in enumerate(sel):
            picks.append((*v, PALETTE.get(n, ['#777', '#999'])[i % 2]))
    return picks


def surviving(ds, keep_t12e=False):
    d = pd.read_csv(REPO / f'output/run_final/construction_performance_table_{ds}.csv')
    m1 = d[d.Task_Idx == 'M1'].copy()
    keep = []
    for k, g in m1.groupby(F):
        if not keep_t12e and k[2] == 'T12e':
            continue
        tr = g[g.run_split == 'train']
        v = tr['S_GNN_step1'].dropna()
        if len(tr) > 0 and int((v == 0).sum()) / len(tr) > ZERO:
            continue
        if v.isna().all():
            continue
        keep.append(k)
    return m1, set(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DISP))
    ap.add_argument('--epoch_dir', default=None, nargs='+',
                    help='One or more directories of epoch-log CSVs; searched in order.')
    ap.add_argument('--select', default='ruleA', choices=['ruleA', 'logged'],
                    help='ruleA = best/worst per node type by GraphRAG composite '
                         '(shared with the GraphRAG figure). logged = whatever is '
                         'in the epoch dirs, the old behaviour.')
    args = ap.parse_args()
    ds = args.dataset
    epoch_dirs = ([Path(d) for d in args.epoch_dir] if args.epoch_dir
                  else [REPO / f'output/run_final/epoch_logs/{ds}'])

    m1, keep = surviving(ds)
    stems = sorted({Path(p).name.rsplit('_', 1)[0]
                    for d in epoch_dirs for p in glob.glob(str(d / '*.csv'))})

    # up to two logged, surviving, non-T12e variants per node type
    chosen, by_node = [], {}
    if args.select == 'ruleA':
        logged = {tuple(st.split('_')[1:]) for st in stems}
        for pick in rule_a(ds, keep):
            v = pick[:3]
            if v in logged:
                chosen.append(pick)
                by_node.setdefault(v[0], []).append(v)
            else:
                print(f'  SKIP {relabel("/".join(v))}: no epoch log in '
                      + ', '.join(str(d) for d in epoch_dirs))
    for s in ([] if args.select == 'ruleA' else stems):
        parts = s.split('_')[1:]
        if len(parts) != 3 or parts[2] == 'T12e' or tuple(parts) not in keep:
            continue
        by_node.setdefault(parts[0], []).append(tuple(parts))
    if args.select != 'ruleA':
        for n in sorted(by_node):
            for i, v in enumerate(by_node[n][:2]):
                chosen.append((*v, PALETTE.get(n, ['#777', '#999'])[i % 2]))

    # The control is deliberately NOT required to pass the zero-exclusion
    # filter. That filter is an analysis gate for dropping degenerate variants,
    # and a no-text control is degenerate by construction -- its whole purpose
    # is to mark the floor. Requiring it to survive would exclude precisely the
    # variants that make the best reference (ArXiv's two logged T6e variants
    # both sit at 100% zero on the pseudo-R2 while scoring 22% Top1).
    control = next((tuple(s.split('_')[1:]) for s in stems if s.endswith('T12e')), None)

    avail_nodes = sorted({k[0] for k in keep})
    missing = [n for n in avail_nodes if n not in by_node]
    print(f'{ds}: node types in data {[relabel(n) for n in avail_nodes]} | '
          f'logged {[relabel(n) for n in sorted(by_node)]}'
          + (f' | MISSING {[relabel(n) for n in missing]}' if missing else ' | complete'))
    print(f'  curves: {[relabel("/".join(c[:3])) for c in chosen]}'
          + (f' + control {relabel("/".join(control))}' if control else ' | NO T6e control available'))

    fig = plt.figure(figsize=(15, 6.5), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.20], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])

    peak = 0.0
    series = [(*c,) for c in chosen] + ([(*control, CONTROL_COLOR)] if control else [])
    for N, E, T, color in series:
        paths = sorted(q for d in epoch_dirs
                       for q in glob.glob(str(d / f'M1_{N}_{E}_{T}_*.csv')))
        curves, is_base = [], (T == 'T12e')
        for i, p in enumerate(paths):
            col = 'test_top1' if 'test_top1' in pd.read_csv(p, nrows=1).columns else 'test_acc'
            d = pd.read_csv(p).set_index('epoch')[col] * 100
            curves.append(d)
            peak = max(peak, float(d.max()))
            ax1.plot(d.index, d.values, color=color, linewidth=1.6, alpha=0.35, zorder=2,
                     label=relabel(f'{N}/{E}/{T}') if (is_base and i == 0) else None)
        if not curves or is_base:
            continue
        mean_curve = pd.concat(curves, axis=1).mean(axis=1)
        ax1.plot(mean_curve.index, mean_curve.values, color=color, linestyle=NODE_STYLE[N],
                 linewidth=5, label=relabel(f'{N}/{E}/{T}'), zorder=3)

    ax1.text(0.5, 1.05, 'Individual Samples = Thin Lines     Mean of Samples = Thick Line',
             transform=ax1.transAxes, ha='center', fontsize=11.5, style='italic')
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('GNN Accuracy', fontsize=15)
    # Adaptive ceiling: ArXiv's N2 (author) variants reach ~93%, which a
    # hardcoded 70% limit silently clipped. Round up to the next 10% with a
    # little headroom, never below 70 so panels stay broadly comparable.
    top = max(70, int((peak + 9) // 10 * 10))
    ax1.set_ylim(0, top)
    ax1.set_yticks(range(0, top + 1, 10))
    ax1.set_yticklabels([f'{v}%' for v in range(0, top + 1, 10)])
    ax1.set_xticks(range(0, 101, 10))
    leg = ax1.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.92,
                     handlelength=2.6, handletextpad=0.8, labelspacing=0.5,
                     borderpad=0.6, alignment='left')
    # handlelength equalises the handle BOX; dashed/dash-dot styles still end
    # mid-gap and read short, so force the swatches solid and uniform. Colour
    # alone identifies each series.
    for h in leg.legend_handles:
        h.set_linestyle('-'); h.set_linewidth(4.0); h.set_alpha(1.0)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(alpha=0.25)
    ax1.tick_params(labelsize=12)

    df = m1[m1[F].apply(tuple, axis=1).isin(keep)].copy()
    df['sample_idx'] = pd.to_numeric(df['sample_idx'], errors='coerce').astype('Int64')
    test_df = df[df.run_split == 'test']
    samples = [s for s in HELD_OUT_SAMPLES if s in test_df.sample_idx.values] or \
              sorted(test_df.sample_idx.unique())[-HELD_OUT_FALLBACK_N:]
    pivot = build_pivot(df, 'Top1', 100.0, samples, compute_row_order(df, samples))
    pivot.index = [relabel(str(i).replace('M1_', '').replace('_', '/')) for i in pivot.index]

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(pivot, ax=ax2, cmap='RdYlGn', linewidths=0.3, linecolor='#e0e0e0',
                cbar=False, yticklabels=True, xticklabels=False,
                annot=True, fmt='.0f', annot_kws={'fontsize': 6.5})
    ax2.set_xlabel(''); ax2.set_ylabel('')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8.5)

    fig.suptitle('GNN Performance on Node Classification', fontsize=19, fontweight='bold', y=1.02)
    plt.tight_layout()
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0 - 0.025, pos.width, pos.height + 0.060])

    out = Path(__file__).resolve().parent / 'artifacts' / f'fig_{ds}_construction_performance_runfinal'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  heatmap rows: {len(pivot)}  saved -> {out.name}.pdf')


if __name__ == '__main__':
    main()
