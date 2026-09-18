#!/usr/bin/env python3
"""Figure 3 (Amazon Sports, main body) and Figure 6 (other four, appendix).

One tree per task per dataset, fit on the train-pool per-variant means from
output/run_final/.  These are ILLUSTRATION trees: depth is capped at three so
the listings fit on a page, which makes them deliberately different objects
from the depth-eight trees behind Table 3.  The reported R^2 is in-sample.

min_samples_leaf is 2, not 3.  With the no-text control excluded each
(Node, Edge) cell holds exactly two variants, so leaf=3 makes a one-hot edge
split unreachable and silently deletes the entire edge axis from every tree.

Writes:  paper/artifacts/fig3_trees_amazon.tex
         paper/artifacts/fig6_trees_appendix.tex

Usage:   python3 paper/make_trees.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AXES, LABEL, out, relabel

import dt_consistency as D
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

MAIN = 'amazon'
APPENDIX = ['arxiv', 'electronics', 'toys', 'history']
MAX_DEPTH, MIN_LEAF = 3, 2
TASKS = [('node_classification', 'node classification'),
         ('graphrag', 'GraphRAG retrieval')]


def fit_one(ds, task):
    """Refit at display depth. The control is always excluded here: a tree
    rooted on 'is this the no-text baseline' is not a construction rule."""
    D.INCLUDE_CONTROL = False
    v = D.TASKS[task](ds)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = enc.fit_transform(v[AXES])
    y = v['train_mean'].to_numpy(float) * 100
    t = DecisionTreeRegressor(criterion='squared_error', max_depth=MAX_DEPTH,
                              min_samples_leaf=MIN_LEAF, random_state=42).fit(X, y)
    txt = export_text(t, feature_names=list(enc.get_feature_names_out(AXES)),
                      decimals=2)
    txt = txt.replace(' <= 0.50', ' == 0').replace(' >  0.50', ' == 1')
    return relabel(txt).rstrip(), len(v), float(t.score(X, y))


def panel(ds, task, title, width):
    body, n, r2 = fit_one(ds, task)
    return (f'\\begin{{minipage}}[t]{{{width}\\textwidth}}\n'
            f'\\centering\\textbf{{{title}}} ($n{{=}}{n}$, $R^2{{=}}{r2:.3f}$)\n'
            f'\\begin{{verbatim}}\n{body}\n\\end{{verbatim}}\n'
            f'\\end{{minipage}}')


def figure3():
    L = ['\\begin{figure}[htbp]', '\\centering', '\\footnotesize']
    for i, (task, name) in enumerate(TASKS):
        L.append(panel(MAIN, task, f'{LABEL[MAIN]}: {name}', 0.46))
        L.append('\\hfill' if i == 0 else '')
    L += [
        '\\caption{Decision trees fit on the same Amazon Sports constructions '
        'for two tasks, both on a 0--100 scale and capped at depth three. '
        '\\textit{Left:} node classification. \\textit{Right:} GraphRAG '
        'retrieval. Both tasks split first on node type and isolate N2, but '
        'they disagree below that split: text fidelity discriminates for '
        'classification, edge type for retrieval.}',
        '\\label{fig:trees-amazon-gnn-graphrag}', '\\end{figure}']
    return '\n'.join(x for x in L if x)


def figure6():
    L = ['\\begin{figure*}[t]', '\\centering', '\\scriptsize']
    for ds in APPENDIX:
        for i, (task, name) in enumerate(TASKS):
            L.append(panel(ds, task, f'{LABEL[ds]}, {name}', 0.48))
            L.append('\\hfill' if i == 0 else '')
        L.append('\\vspace{0.4em}')
    L += [
        '\\caption{Decision trees for the four datasets not shown in '
        'Figure~\\ref{fig:trees-amazon-gnn-graphrag}, fit at the same settings '
        '(no-text control excluded, depth three, two variants per leaf). Left '
        'column is node classification on the normalized GNN score, right '
        'column is GraphRAG retrieval on the RAGAS composite, both scaled to '
        '0--100. $R^2$ is in-sample. Electronics and Toys behave like Amazon '
        'Sports: both tasks split first on node type and isolate N2, and the '
        'penalty is severe for classification but mild for retrieval. ArXiv is '
        'the only dataset where the two tasks disagree at the root. History '
        'splits on edge type for both tasks, but on different edge types.}',
        '\\label{fig:trees-appendix}', '\\end{figure*}']
    return '\n'.join(x for x in L if x)


if __name__ == '__main__':
    for fname, builder in [('fig3_trees_amazon.tex', figure3),
                           ('fig6_trees_appendix.tex', figure6)]:
        p = out(fname)
        p.write_text(builder() + '\n')
        print(f'wrote {p}')
    print('\nroot split per tree:')
    for ds in [MAIN] + APPENDIX:
        for task, name in TASKS:
            body, n, r2 = fit_one(ds, task)
            root = body.split('\n')[0].split('---')[1].split('==')[0].strip()
            print(f'  {LABEL[ds]:14s} {name:20s} n={n:2d} R2={r2:.3f} root={root}')
