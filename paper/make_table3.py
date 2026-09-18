#!/usr/bin/env python3
"""Table 3 -- decision-tree consistency, both tasks.

Runs code/dt_consistency.py to produce the matrices, then formats them.
The no-text control is KEPT in the node-classification variant set so the
counts match Table 4 and Figure 4 exactly.  That inflates mean within-dataset
rho from 0.93 to 0.97 and removes every negative cross-dataset transfer on
the GNN side; --no-control reproduces the conservative version.

Writes:  paper/artifacts/table3_consistency.tex
Usage:   python3 paper/make_table3.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS, LABEL, REPO, RUN, SHORT, out

import pandas as pd

ALPHA = 0.05          # Bonferroni-corrected over the 25 cells of each panel
DECIMALS = 2
PANELS = [('node_classification', 'GNN-Based Decision Tree Consistency'),
          ('graphrag', 'GraphRAG-Based Decision Tree Consistency')]


def refresh(analysis: Path, control: bool):
    cmd = [sys.executable, str(REPO / 'code' / 'dt_consistency.py'),
           '--out', str(analysis)] + (['--include_control'] if control else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    print(f'  refreshed {analysis.name}')


def panel(analysis: Path, task: str, heading: str) -> str:
    rho = pd.read_csv(analysis / f'{task}_rho.csv', index_col=0)
    p = pd.read_csv(analysis / f'{task}_p.csv', index_col=0)
    summ = pd.read_csv(analysis / f'{task}_summary.csv').set_index('dataset')
    a = ALPHA / len(DATASETS) ** 2

    L = [f'\\textbf{{{heading}}}', '', '\\begin{tabular}[t]{lccccc}', '\\toprule',
         '$\\rho$ & ' + ' & '.join(SHORT[d] for d in DATASETS) + ' \\\\', '\\midrule']
    for r in DATASETS:
        cells = []
        for c in DATASETS:
            v = f'{rho.loc[r, c]:+.{DECIMALS}f}'
            v += '*' if p.loc[r, c] < a else '\\phantom{*}'
            cells.append(f'\\textbf{{{v}}}' if r == c else v)
        L.append(f'{SHORT[r]:<6}& ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}%', '\\hspace{2em}',
          '\\begin{tabular}[t]{lrcc}', '\\toprule',
          'Dataset & $n$ & Within & Cross \\\\', '\\midrule']
    for d in DATASETS:
        s = summ.loc[d]
        L.append(f'{LABEL[d]:<14}& {int(s.n_variants):<3}& '
                 f'\\textbf{{{s.ted_within:.{DECIMALS}f}}} & '
                 f'{s.ted_cross:.{DECIMALS}f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-control', dest='control', action='store_false')
    ap.add_argument('--skip-refresh', action='store_true')
    a = ap.parse_args()

    analysis = RUN / 'analysis' / ('dt_consistency_with_control' if a.control
                                   else 'dt_consistency')
    if not a.skip_refresh:
        refresh(analysis, a.control)

    body = [
        '\\begin{table}[htbp]', '\\centering',
        '\\renewcommand{\\arraystretch}{1.25}', '\\small',
        '\\setlength{\\tabcolsep}{3.7pt}',
        '\\caption{Decision trees trained on one dataset transfer to other '
        'datasets to varying degrees, evaluated separately for GNN and '
        'GraphRAG settings. \\textit{Left:} cross-dataset Spearman $\\rho$ '
        '(rows = training dataset, columns = target dataset; diagonal = '
        f'within-dataset consistency; $^{{*}}p<{ALPHA}$ after Bonferroni '
        'correction over the 25 cells of each panel). \\textit{Right:} '
        'normalized tree edit distance, within dataset and averaged over the '
        'four cross-dataset pairs (lower = more structurally similar). '
        'Variant counts match Table~\\ref{tab:timing}.}',
        '\\label{tab:dt_consistency}', '']
    for i, (task, heading) in enumerate(PANELS):
        body += ['\\vspace{0.5em}' if i == 0 else '\\vspace{1em}',
                 panel(analysis, task, heading), '']
    body.append('\\end{table}')

    p = out('table3_consistency.tex')
    p.write_text('\n'.join(body) + '\n')
    print(f'wrote {p}')
