#!/usr/bin/env python3
"""Table 4 -- cost of scoring the construction space vs one full-scale model.

Runs code/build_table5_timing.py (which reads the measured wall times and the
stability curve, and picks each dataset's plateau as the smallest budget where
rho stays above 0.80), then formats the paper's column layout.

Whole-space cost = total scoring at the plateau / one full-scale run.  GNN uses
the plateau; GraphRAG uses its full budget, because its rho is still improving
at 300 questions.

Writes:  paper/artifacts/table4_timing.tex
Usage:   python3 paper/make_table4.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LABEL, REPO, RUN, out

import pandas as pd

ORDER = ['arxiv', 'amazon', 'history', 'electronics', 'toys']
GNN_FULL_N = 150


def refresh():
    r = subprocess.run([sys.executable, str(REPO / 'code' / 'build_table5_timing.py')],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)


def build() -> str:
    raw = pd.read_csv(RUN / 'analysis' / 'timing_table_raw.csv').set_index('dataset')
    pla = pd.read_csv(RUN / 'analysis' / 'timing_table_plateau.csv').set_index('dataset')

    rows, tot = [], {k: 0.0 for k in
                     ['gnn_tot', 'gra_tot', 'gnn_base', 'gra_base', 'plat', 'ng', 'nr']}
    for ds in ORDER:
        r, p = raw.loc[ds], pla.loc[ds]
        plat_var = r.mean_gnn_s / GNN_FULL_N * p.gnn_plateau_n
        space_g = r.n_gnn * plat_var / r.base_gnn_s
        space_r = r.total_gra_s / r.base_gra_s
        rows.append(
            f'{LABEL[ds].replace("Amazon Sports", "Amazon"):<12}& {int(r.n_gnn)} & {int(r.n_gra)} '
            f'& {r.mean_gnn_s/60:.1f} & {r.mean_gra_s/60:.1f} '
            f'& {r.total_gnn_s/3600:.2f} & {r.total_gra_s/3600:.2f} '
            f'& {r.base_gnn_s/60:.2f} & {r.base_gra_s/60:.2f} '
            f'& $\\mathbf{{{space_g:.1f}\\times}}$ & ${space_r:.1f}\\times$ \\\\')
        for k, v in [('gnn_tot', r.total_gnn_s), ('gra_tot', r.total_gra_s),
                     ('gnn_base', r.base_gnn_s), ('gra_base', r.base_gra_s),
                     ('plat', r.n_gnn * plat_var), ('ng', r.n_gnn), ('nr', r.n_gra)]:
            tot[k] += v

    total = (f'\\textbf{{Total}} & \\textbf{{{int(tot["ng"])}}} & \\textbf{{{int(tot["nr"])}}} '
             f'& \\textbf{{{tot["gnn_tot"]/tot["ng"]/60:.1f}}} & \\textbf{{{tot["gra_tot"]/tot["nr"]/60:.1f}}} '
             f'& \\textbf{{{tot["gnn_tot"]/3600:.2f}}} & \\textbf{{{tot["gra_tot"]/3600:.2f}}} '
             f'& \\textbf{{{tot["gnn_base"]/60:.2f}}} & \\textbf{{{tot["gra_base"]/60:.2f}}} '
             f'& $\\mathbf{{{tot["plat"]/tot["gnn_base"]:.1f}\\times}}$ '
             f'& $\\mathbf{{{tot["gra_tot"]/tot["gra_base"]:.1f}\\times}}$ \\\\')

    rho_lo, rho_hi = pla.gnn_plateau_rho.min(), pla.gnn_plateau_rho.max()
    full_lo = min(raw.loc[d, 'total_gnn_s'] / raw.loc[d, 'base_gnn_s'] for d in ORDER)
    full_hi = max(raw.loc[d, 'total_gnn_s'] / raw.loc[d, 'base_gnn_s'] for d in ORDER)

    return '\n'.join([
        '\\begin{table*}[h]', '\\centering',
        '\\renewcommand{\\arraystretch}{1.25}',
        '\\caption{Cost of scoring the construction space relative to one '
        'full-scale model. \\textbf{Scoring/variant} is the wall time to score '
        'one variant on the proxy evaluation: $150$ subsets of $1$k nodes for '
        'GNNs, $300$ questions for GraphRAG ($233$ for History). \\textbf{Total '
        'scoring} sums over all valid variants. \\textbf{Full-scale run} is one '
        'unguided variant at full data scale. \\textbf{Whole-space cost} is the '
        'cost of scoring every variant in multiples of that run. GNN uses the '
        f'five-subset plateau, where $\\rho$ reaches ${rho_lo:.2f}$ to ${rho_hi:.2f}$; '
        f'the full $150$-subset budget costs ${full_lo:.0f}$ to ${full_hi:.0f}\\times$. '
        'GraphRAG uses its full budget, since its $\\rho$ is still improving '
        'there. Variant counts include the no-text control, which GraphRAG '
        'cannot run.}',
        '\\vspace{5pt}', '\\label{tab:timing}',
        '\\setlength{\\tabcolsep}{3.6pt}', '\\small',
        '\\begin{tabular}{@{}l cc cc cc cc cc@{}}', '\\toprule',
        ' & \\multicolumn{2}{c}{Variants} & \\multicolumn{2}{c}{Scoring/variant (min)}',
        ' & \\multicolumn{2}{c}{Total scoring (hr)} & \\multicolumn{2}{c}{Full-scale run (min)}',
        ' & \\multicolumn{2}{c}{Whole-space cost} \\\\',
        '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}',
        '\\cmidrule(lr){8-9} \\cmidrule(lr){10-11}',
        'Dataset & GNN & RAG & GNN & RAG & GNN & RAG & GNN & RAG & GNN & RAG \\\\',
        '\\midrule', *rows, '\\midrule', total,
        '\\bottomrule', '\\end{tabular}', '\\end{table*}'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-refresh', action='store_true')
    if not ap.parse_args().skip_refresh:
        refresh()
    p = out('table4_timing.tex')
    p.write_text(build() + '\n')
    print(f'wrote {p}')
