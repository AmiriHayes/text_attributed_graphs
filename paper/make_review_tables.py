#!/usr/bin/env python3
"""Three tables the manuscript is missing, as drop-in LaTeX.

  table5_ablation.tex     what the tree contributes (no-tree floor, LOVO ceiling)
  table6_payoff.tex       what following the tree buys a practitioner
  table7_full_scale.tex   does the subset ranking predict full-scale performance

Sources:
  output/run_final/analysis/tree_ablation/tree_ablation.csv   (code/tree_ablation.py)
  output/run_final/analysis/full_scale/full_scale_validation.csv
                                                     (code/full_scale_validation.py)

Usage:  python3 paper/make_review_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, RUN                                # noqa: E402

ART = REPO / 'paper' / 'artifacts'
LABEL = {'history': 'History', 'amazon': 'Sports', 'arxiv': 'ArXiv',
         'electronics': 'Electronics', 'toys': 'Toys'}
ORDER = ['history', 'amazon', 'arxiv', 'electronics', 'toys']
TASK = {'node_classification': 'Node classification', 'graphrag': 'GraphRAG retrieval'}


def _f(x, nd=2, dash='--'):
    return dash if pd.isna(x) else f'{x:.{nd}f}'


def ablation(df: pd.DataFrame) -> str:
    body = []
    for task in ('node_classification', 'graphrag'):
        t = df[df.task == task].set_index('dataset')
        body.append(rf'\multicolumn{{5}}{{c}}{{\textit{{{TASK[task]}}}}} \\')
        body.append(r'\midrule')
        for ds in ORDER:
            r = t.loc[ds]
            body.append(
                rf'{LABEL[ds]} & {int(r.n_variants)} & {_f(r.rho_no_tree, 3)} & '
                rf'{_f(r.rho_tree, 3)} & {_f(r.rho_lovo, 3)} \\')
        m = t[['rho_no_tree', 'rho_tree', 'rho_lovo']].mean()
        body.append(rf'\textbf{{Mean}} & & \textbf{{{m.rho_no_tree:.3f}}} & '
                    rf'\textbf{{{m.rho_tree:.3f}}} & \textbf{{{m.rho_lovo:.3f}}} \\')
        if task == 'node_classification':
            body.append(r'\midrule')
    return r"""\begin{table}[htbp]
\centering
\small
\caption{What the decision tree contributes. \textbf{No tree} ranks constructions
by their raw train-pool mean score. \textbf{Tree} is the estimator reported in
Table~\ref{tab:dt_consistency}, evaluated on the same variants it was fit on.
\textbf{LOVO} refits the tree with one construction held out and predicts that
construction, the only setting in which the tree ranks a variant it has not
scored. All three are Spearman $\rho$ against the held-out test-pool means. The
tree matches the no-tree floor, so the headline $\rho$ measures the
reproducibility of the proxy scores; the tree's contribution is compression of
the score table into readable rules, and its generalization to unscored
constructions is given by the LOVO column.}
\label{tab:tree_ablation}
\vspace{4pt}
\begin{tabular}{lrccc}
\toprule
Dataset & $n$ & No tree & Tree & LOVO \\
\midrule
""" + '\n'.join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def payoff(df: pd.DataFrame) -> str:
    body = []
    for task in ('node_classification', 'graphrag'):
        t = df[df.task == task].set_index('dataset')
        body.append(rf'\multicolumn{{6}}{{c}}{{\textit{{{TASK[task]}}}}} \\')
        body.append(r'\midrule')
        for ds in ORDER:
            r = t.loc[ds]
            body.append(
                rf'{LABEL[ds]} & {_f(r.blind_mean)} & {_f(r.default_N1_T6a)} & '
                rf'{_f(r.taco_pick)} & {_f(r.oracle)} & {_f(r.pct_of_oracle_gain, 1)}\% \\')
        body.append(r'\midrule' if task == 'node_classification' else '')
    return r"""\begin{table}[htbp]
\centering
\small
\caption{What following TACO-Graph buys. All columns are test-pool scores on a
0--100 scale. \textbf{Blind} is the mean over all valid constructions, what a
practitioner gets by picking one arbitrarily. \textbf{Default} is the untuned
choice of primary entity with the richest text (N1/T6a). \textbf{TACO} is the
construction the tree ranks first. \textbf{Oracle} is the best construction in
hindsight. The last column is the fraction of the achievable gain over a blind
pick that TACO recovers. The default is a strong baseline and TACO matches or
beats it on nine of ten dataset--task pairs; the exception is Sports node
classification, and the largest margin is ArXiv, where the winning construction
is the non-obvious secondary entity.}
\label{tab:payoff}
\vspace{4pt}
\begin{tabular}{lccccc}
\toprule
Dataset & Blind & Default & TACO & Oracle & Gain recovered \\
\midrule
""" + '\n'.join(x for x in body if x) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def full_scale(df: pd.DataFrame) -> str:
    body = []
    for ds in ORDER:
        if ds not in set(df.dataset):
            continue
        r = df[df.dataset == ds].iloc[0]
        guard = int(r.n_over_pipeline_guard)
        star = r'$^{\dagger}$' if bool(r.top_pick_over_pipeline_guard) else ''
        body.append(
            rf'{LABEL[ds]} & {int(r.n_variants)} & {guard}{star} & '
            rf'{_f(r.get("rho_train_mean_vs_full"), 3)} & '
            rf'{_f(r.get("rho_tree_vs_full"), 3)} & '
            rf'{_f(r.get("pct_gain_tree"), 1)}\% & '
            rf'{_f(r.get("mean_subset_score"))} & {_f(r.get("mean_full_score"))} \\')
    return r"""\begin{table}[htbp]
\centering
\small
\caption{Does the subset ranking predict full-scale performance? Every variant
was additionally trained once on the complete held-out pool (20{,}775--50{,}000
rows rather than 1{,}000), with identical architecture and hyperparameters.
\textbf{Over guard} counts variants whose full-scale graph exceeds the
pipeline's own $200{,}000$-edge degeneracy limit, so the proxy ranks
constructions that cannot be trained at full scale; $^{\dagger}$ marks datasets
where the top-ranked construction is one of them. $\rho$ columns are Spearman
against the full-scale score. \textbf{Gain} is the share of the oracle
improvement over a blind pick that the tree's choice recovers \emph{at full
scale}. The last two columns show that subset scoring shifts the absolute level
substantially while preserving order.}
\label{tab:full_scale}
\vspace{4pt}
\begin{tabular}{lrrccccc}
\toprule
 & \multicolumn{2}{c}{Variants} & \multicolumn{2}{c}{$\rho$ vs full scale}
 & & \multicolumn{2}{c}{Mean score} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){7-8}
Dataset & All & Over guard & Subset mean & Tree & Gain & Subset & Full \\
\midrule
""" + '\n'.join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    abl = RUN / 'analysis' / 'tree_ablation' / 'tree_ablation.csv'
    fsv = RUN / 'analysis' / 'full_scale' / 'full_scale_validation.csv'

    if abl.exists():
        d = pd.read_csv(abl)
        (ART / 'table5_ablation.tex').write_text(ablation(d))
        (ART / 'table6_payoff.tex').write_text(payoff(d))
        print(f'wrote {ART / "table5_ablation.tex"}')
        print(f'wrote {ART / "table6_payoff.tex"}')
    else:
        print(f'missing {abl} -- run code/tree_ablation.py')

    if fsv.exists():
        (ART / 'table7_full_scale.tex').write_text(full_scale(pd.read_csv(fsv)))
        print(f'wrote {ART / "table7_full_scale.tex"}')
    else:
        print(f'missing {fsv} -- run code/full_scale_score.py then '
              'code/full_scale_validation.py')


if __name__ == '__main__':
    main()
