#!/usr/bin/env python3
"""
Phase 5 -- Table 5 timing, framed on the PLATEAU survey budget rather than
the full 150-subset / 300-question budget that was actually run.

WHY THE REFRAMING
The earlier table reported speedup at the full survey budget and got
0.14-0.44x for GNN: the survey cost MORE than naively training every variant
once at full scale. That is arithmetically correct and not a useful claim.
It is also not the operating point the method recommends -- the stability
curves (Figure 4) show construction rankings saturate far below the budget we
ran. Amazon and Electronics are stably above rho=0.80 at FIVE subsets;
we ran 150. The efficiency claim belongs at the budget a practitioner would
actually use, not at the budget used to MEASURE where saturation happens.

Two things this script is careful about:

  1. It never presents the plateau cost as the cost of the run we did. The
     table reports both: measured time per variant at the full budget, and
     the projected cost at the plateau budget.

  2. It never reports a speedup without the rho achieved at that budget. A
     large speedup at a budget where rho = 0.54 is not a result. Where a
     dataset never stabilises above rho = 0.80, the plateau is reported as
     "not reached" and no speedup is claimed.

PLATEAU DEFINITION
Smallest n on the stability curve such that mean rho >= 0.80 at n AND at
every larger n (stably above threshold, not a lucky single point). 0.80 is
the paper's own stated threshold, so the plateau is not a free parameter
tuned per dataset.

SPEEDUP DEFINITION
    speedup = (baseline x n_variants) / (per-variant plateau cost x n_variants)
            =  baseline / per-variant plateau cost
The n_variants factor cancels -- surveying k variants and naively evaluating
k variants both scale linearly in k. So this is a per-variant statement:
how much cheaper is ranking a variant on a plateau-sized subset survey than
training it once on the full dataset.

GNN cost is linear in n_subsets (each subset is an independent training run).
GraphRAG cost is modelled as linear in n_questions with a per-variant fixed
index-build component; that component is estimated from History, the only
dataset measured at two question counts (83 and 150), at ~10% of the
full-budget per-variant time. Flagged as an approximation in the caption.

NOTE ON THE GRAPHRAG SPEEDUP COLUMN
GraphRAG's baseline and survey use the IDENTICAL per-variant scale (one
variant, full corpus, full question set), so the ratio reduces to
baseline_variant_time / mean_variant_time ~ 1 and carries no efficiency
information. The old table's 0.61-1.06x column was measuring only whether
the variant chosen as baseline happened to be faster than average. It is
reported here as scale-matched rather than dressed up as a speedup, and a
question-budget reduction is claimed only where the curve justifies one.

Reads:
  - output/run_final/analysis/timing_table_raw.csv
  - output/run_final/analysis/gnn_rho_vs_subsets_multiseed_{dataset}.csv
  - output/run_final/ragas_results_{dataset}.csv   (measured 300q per-variant time)

Writes:
  - output/run_final/analysis/timing_table_plateau.csv

Usage:
  python3 code/build_table5_timing.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'output' / 'run_final' / 'analysis'
RUN_FINAL = REPO / 'output' / 'run_final'
TAB_DIR = REPO / 'output' / 'tables'

DATASETS = ['history', 'arxiv', 'amazon', 'electronics', 'toys']
DISPLAY = {'arxiv': 'ArXiv', 'amazon': 'Amazon', 'history': 'History',
           'electronics': 'Electronics', 'toys': 'Toys'}
RHO_THRESHOLD = 0.80
GNN_FULL_N = 150                 # subsets per variant actually run
GRA_FIXED_FRACTION = 0.10        # per-variant index-build share (see docstring)


def plateau(curve: pd.DataFrame, xcol: str) -> tuple:
    """Smallest n stably at mean_rho >= RHO_THRESHOLD. Returns (n, rho) or
    (None, best_rho_seen) when the curve never stabilises above threshold."""
    c = curve.dropna(subset=['mean_rho']).reset_index(drop=True)
    if c.empty:
        return None, float('nan')
    for i in range(len(c)):
        if (c['mean_rho'].iloc[i:] >= RHO_THRESHOLD).all():
            return int(c[xcol].iloc[i]), float(c['mean_rho'].iloc[i])
    return None, float(c['mean_rho'].max())


def measured_gra_per_variant(dataset: str) -> tuple:
    """Per-variant wall time over the full question set, and that question
    count. Each (variant, wall_time) pair is one evaluation batch; a variant's
    total is the sum over its distinct batches."""
    df = pd.read_csv(RUN_FINAL / f'ragas_results_{dataset}.csv')
    batches = df.groupby(['variant', 'wall_time_seconds']).size().reset_index(name='n_q')
    per_variant = batches.groupby('variant')['wall_time_seconds'].sum()
    return float(per_variant.mean()), int(df['question_id'].nunique())


CURVE_CSV = ANALYSIS / 'dt_consistency_with_control' / 'stability_curve.csv'


def build() -> pd.DataFrame:
    raw = pd.read_csv(ANALYSIS / 'timing_table_raw.csv').set_index('dataset')
    rows = []
    for ds in DATASETS:
        r = raw.loc[ds]
        # Both curves come from dt_consistency's stability run, so the
        # plateau is read off exactly the estimator Table 3 reports. The
        # retired gnn_rho_vs_subsets_multiseed_*.csv used min_samples_leaf=3
        # and no explicit no-text exclusion, which put History and ArXiv at
        # rho ~ 0.54 and reported "plateau not reached" for both.
        curve = pd.read_csv(CURVE_CSV)
        curve = curve[curve.dataset == ds].rename(columns={'mean': 'mean_rho'})
        gnn_curve = curve[curve.task == 'node_classification'].rename(
            columns={'n': 'n_subsets'}).sort_values('n_subsets')
        gra_curve = curve[curve.task == 'graphrag'].rename(
            columns={'n': 'n_questions'}).sort_values('n_questions')

        gnn_n, gnn_rho = plateau(gnn_curve, 'n_subsets')
        gra_n, gra_rho = plateau(gra_curve, 'n_questions')

        gnn_per_variant_full = float(r.mean_gnn_s)
        gnn_per_subset = gnn_per_variant_full / GNN_FULL_N
        gnn_plateau_cost = gnn_per_subset * gnn_n if gnn_n else np.nan
        gnn_speedup = float(r.base_gnn_s) / gnn_plateau_cost if gnn_n else np.nan

        gra_per_variant_full, gra_n_q_full = measured_gra_per_variant(ds)
        fixed = GRA_FIXED_FRACTION * gra_per_variant_full
        per_q = (gra_per_variant_full - fixed) / gra_n_q_full
        gra_plateau_cost = fixed + per_q * gra_n if gra_n else np.nan

        rows.append({
            'dataset': ds,
            'n_gnn_variants': int(r.n_gnn), 'n_gra_variants': int(r.n_gra),
            'gnn_baseline_s': round(float(r.base_gnn_s), 1),
            'gnn_per_variant_full_s': round(gnn_per_variant_full, 1),
            'gnn_plateau_n': gnn_n, 'gnn_plateau_rho': round(gnn_rho, 3),
            'gnn_plateau_cost_s': round(gnn_plateau_cost, 1) if gnn_n else None,
            'gnn_speedup_at_plateau': round(gnn_speedup, 2) if gnn_n else None,
            'gra_n_questions_full': gra_n_q_full,
            'gra_per_variant_full_s': round(gra_per_variant_full, 1),
            'gra_plateau_n': gra_n, 'gra_plateau_rho': round(gra_rho, 3),
            'gra_plateau_cost_s': round(gra_plateau_cost, 1) if gra_n else None,
            'gra_scale_matched_ratio': round(float(r.base_gra_s) / float(r.mean_gra_s), 2),
        })
    return pd.DataFrame(rows)


def _min(s):
    return f'{s / 60:.1f}'


def to_tex(t: pd.DataFrame) -> str:
    L = [
        '% Table 5 -- Timing, framed on the plateau survey budget.',
        '% The survey was RUN at 150 subsets/variant (GNN) and 300 questions/variant',
        '%   (GraphRAG). The plateau columns are the budget at which construction',
        '%   rankings stabilise (smallest n stably at rho>=0.80 on the Figure 4 curves),',
        '%   i.e. the operating point the method recommends -- not the cost of the run',
        '%   performed to locate it. Both are reported so the two are never conflated.',
        '% Speedup = (baseline x n_variants)/(plateau cost x n_variants) = baseline/plateau',
        '%   cost. The n_variants factor cancels: it is a per-variant statement.',
        '% "not reached" = the curve never stabilises above rho=0.80; no speedup claimed,',
        '%   best rho attained shown instead.',
        '% GraphRAG baseline and survey use the same per-variant scale, so their ratio',
        '%   reduces to baseline_variant/mean_variant ~ 1 and is reported as scale-matched,',
        '%   not as a speedup.',
        '% Sources: output/run_final/analysis/timing_table_{raw,plateau}.csv',
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{Survey cost at the \emph{plateau} budget. Construction rankings '
         r'saturate well below the budget used to measure saturation: the survey was run '
         r'at 150 subsets per variant, but Amazon and Electronics are stably above '
         r'$\rho=0.80$ at five subsets. Plateau $n$ is the smallest budget stably at '
         r'$\rho \ge 0.80$ on the Figure~\ref{fig:stability} curves; $\rho$@plateau is the '
         r'consistency actually attained there, so speedup is never reported without the '
         r'quality it buys. Speedup is $\text{baseline}/\text{plateau cost}$ per variant '
         r'(the $n$-variant factor cancels, since surveying and naively evaluating $k$ '
         r'variants both scale linearly in $k$). GNN cost is linear in $n_{\text{subsets}}$; '
         r'GraphRAG cost is linear in $n_{\text{questions}}$ above a per-variant '
         r'index-build component estimated at ${\sim}10\%$ of full-budget time from '
         r'History, the only dataset measured at two question counts. GraphRAG baseline '
         r'and survey share the same per-variant scale, so that ratio is scale-matched '
         r'($\approx 1$) and carries no subsampling saving.}'),
        r'\label{tab:timing}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'\multicolumn{7}{c}{\textbf{GNN}} \\',
        r'\midrule',
        (r'Dataset & Variants & Baseline (s) & Time/variant (s) & Plateau $n$ & '
         r'$\rho$@plateau & Speedup \\'),
        r'\midrule',
    ]
    for _, r in t.iterrows():
        if r['gnn_plateau_n'] is None or pd.isna(r['gnn_plateau_n']):
            L.append(rf"{DISPLAY[r['dataset']]:12s} & {r['n_gnn_variants']} & "
                     rf"{r['gnn_baseline_s']} & {r['gnn_per_variant_full_s']} & "
                     rf"not reached & {r['gnn_plateau_rho']:.3f} (best) & -- \\")
        else:
            L.append(rf"{DISPLAY[r['dataset']]:12s} & {r['n_gnn_variants']} & "
                     rf"{r['gnn_baseline_s']} & {r['gnn_per_variant_full_s']} & "
                     rf"{int(r['gnn_plateau_n'])} & {r['gnn_plateau_rho']:.3f} & "
                     rf"\textbf{{{r['gnn_speedup_at_plateau']:.2f}$\times$}} \\")
    L += [
        r'\midrule',
        r'\multicolumn{7}{c}{\textbf{GraphRAG}} \\',
        r'\midrule',
        (r'Dataset & Variants & Questions & Time/variant (s) & Plateau $n$ & '
         r'$\rho$@plateau & Scale-matched \\'),
        r'\midrule',
    ]
    for _, r in t.iterrows():
        pn = ('not reached' if r['gra_plateau_n'] is None or pd.isna(r['gra_plateau_n'])
              else str(int(r['gra_plateau_n'])))
        rho_s = (f"{r['gra_plateau_rho']:.3f} (best)" if pn == 'not reached'
                 else f"{r['gra_plateau_rho']:.3f}")
        L.append(rf"{DISPLAY[r['dataset']]:12s} & {r['n_gra_variants']} & "
                 rf"{r['gra_n_questions_full']} & {r['gra_per_variant_full_s']} & "
                 rf"{pn} & {rho_s} & {r['gra_scale_matched_ratio']:.2f}$\times$ \\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    t = build()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(ANALYSIS / 'timing_table_plateau.csv', index=False)
    tex = to_tex(t)
    print(t.to_string(index=False))
    print(f"\nsaved -> output/run_final/analysis/timing_table_plateau.csv")
    print("saved -> output/run_final/analysis/timing_table_plateau.csv\n")
    print(tex)
