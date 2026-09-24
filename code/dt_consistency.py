#!/usr/bin/env python3
"""
Decision-tree consistency: held-out rho and tree edit distance.

A clean reimplementation with no dependency on the retired M1-M6 task
taxonomy.  Each cell of every table is computed for ONE task and ONE dataset
at a time.  Node-classification scores never mix with GraphRAG scores, and
one dataset's scores never mix with another's.

Per (task, dataset):
  1. Split the evaluation units evenly into a train pool and a test pool.
       node classification : 150 subsets  -> 75 train / 75 test
       graphrag retrieval  : 300 questions -> 150 train / 150 test
     Both splits are fixed upstream and read, never recomputed here.
  2. For each surviving variant, average its score over the train pool
     (train_mean) and over the test pool (test_mean).  Two vectors indexed by
     variant.
  3. Fit a regression tree on one-hot (Node_Idx, Edge_Idx, Text_Idx) with
     train_mean as target.  The tree never sees the test pool.
  4. rho = Spearman(train_tree.predict(variants), test_mean).
  5. Fit a second tree, same hyperparameters, with test_mean as target.
     TED = normalized tree edit distance between the two trees.

Cross-dataset cells reuse steps 3-5 with the tree from dataset A applied to
dataset B's variants.  They are the interpretability anchor for TED: a
within-dataset TED only means something relative to the TED between trees
fit on genuinely different datasets.

Every tree in this file uses identical hyperparameters (see TREE_KWARGS) so
that every number in every table is on the same footing.

  min_samples_leaf is 2, not the 3 used by the legacy pipeline.  With the
  no-text baseline excluded, each (Node_Idx, Edge_Idx) cell holds exactly two
  variants, so min_samples_leaf=3 makes a one-hot edge split unreachable and
  silently deletes the entire edge axis from every tree.

Reads:
  output/run_final/construction_performance_table_{dataset}.csv
  output/run_final/ragas_results_{dataset}.csv
  output/run_final/question_split_{dataset}.csv

Writes (to output/run_final/analysis/dt_consistency/):
  {task}_rho.csv, {task}_p.csv, {task}_ted.csv, {task}_summary.csv
  ted_random_baseline.csv
  table3_dt_consistency.tex

Usage:
  python3 code/dt_consistency.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / 'output' / 'run_final'
OUT = RUN / 'analysis' / 'dt_consistency'

DATASETS = ['history', 'amazon', 'arxiv', 'electronics', 'toys']
LABEL = {'history': 'History', 'amazon': 'Sports', 'arxiv': 'ArXiv',
         'electronics': 'Electronics', 'toys': 'Toys'}
SHORT = {'history': 'Hist', 'amazon': 'Sports', 'arxiv': 'ArXiv',
         'electronics': 'Elec', 'toys': 'Toys'}

AXES = ['Node_Idx', 'Edge_Idx', 'Text_Idx']
TREE_KWARGS = dict(criterion='squared_error', max_depth=8,
                   min_samples_leaf=2, random_state=42)

# The no-text control is a baseline, not a construction choice, so it is
# excluded by name rather than left to the numeric filter below.  Its scores
# are near zero but not always exactly zero, and when it survives the filter
# the tree roots on "is this the control", which is not a construction rule.
NO_TEXT_CONTROL = 'T12e'

# When True, the no-text control is kept in the node-classification variant set
# and the zero-score filter is not applied, so the variant counts match the
# cost table exactly (it is what a practitioner actually enumerates and pays
# to score).  The control's scores sit at the majority-class floor and are
# trivially rank-separable, which RAISES mean within-dataset rho from 0.93 to
# 0.97 -- report both if this is on.  GraphRAG is unaffected either way: a
# no-text graph cannot answer questions, so those variants were never run.
INCLUDE_CONTROL = False

# A variant whose train-pool scores are >95% exact zeros carries no signal and
# is dropped before any tree is fit.
ZERO_FRACTION_LIMIT = 0.95

# When True, edge rules that build byte-identical graphs are collapsed to one
# before any tree is fit.  code/detect_duplicate_edge_rules.py finds E10b and
# E10c colliding on every (node type) of amazon, arxiv, electronics and toys;
# only history distinguishes them.  A collision is not cosmetic: the two
# one-hot columns are perfectly collinear, so a split between them is training
# noise presented as a construction rule, and the duplicated pair contributes a
# guaranteed-consistent point to every rank correlation.  Off by default so the
# published numbers are reproduced unchanged; turn it on to see what the space
# looks like once each distinct graph is counted once.
DEDUPE_EDGES = False
DUPLICATES_CSV = RUN / 'analysis' / 'edge_rule_duplicates.csv'

# Legacy column value identifying node-classification rows in the raw table.
# It is a row filter only.  Task is never a feature and never a grouping key.
NODE_CLASSIFICATION_ROWS = 'M1'

N_RANDOM_SEEDS = 100


# ── variant tables ────────────────────────────────────────────────────────────

def _pool_means(df: pd.DataFrame, unit_split: pd.Series, score: str) -> pd.DataFrame:
    """Per-variant mean over the train pool and over the test pool."""
    df = df.assign(_pool=unit_split)
    df = df[df['_pool'].isin(['train', 'test'])]

    if not INCLUDE_CONTROL:
        df = df[df['Text_Idx'] != NO_TEXT_CONTROL]

        keep = []
        for axes, grp in df.groupby(AXES):
            tr = grp[grp['_pool'] == 'train'][score]
            valid = tr.dropna()
            if len(valid) == 0:
                continue
            if (valid == 0.0).sum() / len(tr) > ZERO_FRACTION_LIMIT:
                continue
            keep.append(axes)
        df = df[df[AXES].apply(tuple, axis=1).isin(set(keep))]

    out = (df.pivot_table(index=AXES, columns='_pool', values=score, aggfunc='mean')
             .rename(columns={'train': 'train_mean', 'test': 'test_mean'})
             .dropna(subset=['train_mean', 'test_mean'])
             .reset_index())
    out['train_mean'] *= 100.0
    out['test_mean'] *= 100.0
    return out[AXES + ['train_mean', 'test_mean']]


def _drop_duplicate_edges(out: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Collapse edge rules that build the same graph, per (dataset, node type).

    Within a colliding group the last rule by name is kept, which retains E10c
    over E10b: edge_factory._build_scalar_gt builds co-participation edges, so
    E10c ('participation') is the label that matches what the code does.
    """
    if not DEDUPE_EDGES or not DUPLICATES_CSV.exists():
        return out
    dup = pd.read_csv(DUPLICATES_CSV)
    dup = dup[dup['dataset'] == ds]
    drop = set()
    for _, r in dup.iterrows():
        rules = sorted(str(r['duplicate_rules']).split('+'))
        for rule in rules[:-1]:
            drop.add((r['node_type'], rule))
    if not drop:
        return out
    mask = ~out.apply(lambda row: (row['Node_Idx'], row['Edge_Idx']) in drop, axis=1)
    return out[mask].reset_index(drop=True)


def node_classification_variants(ds: str) -> pd.DataFrame:
    """150 subsets, 75 train / 75 test, scored by the normalized GNN score."""
    df = pd.read_csv(RUN / f'construction_performance_table_{ds}.csv')
    df = df[df['Task_Idx'] == NODE_CLASSIFICATION_ROWS].copy()
    # No fallback: S_GNN_step1 is a 0-1 pseudo-R2 while normalized_score runs
    # 0-70 on the same rows, so silently substituting one changes every
    # published number by roughly two orders of magnitude.
    if 'S_GNN_step1' not in df.columns:
        raise KeyError(
            f"{ds}: construction table has no 'S_GNN_step1' column "
            f"(found: {sorted(df.columns)[:8]}...). Re-run "
            "code/experiment_runner.py; do not substitute 'normalized_score', "
            "which is on a different scale.")
    score = 'S_GNN_step1'
    return _drop_duplicate_edges(_pool_means(df, df['run_split'], score), ds)


def graphrag_variants(ds: str) -> pd.DataFrame:
    """300 questions, 150 train / 150 test, scored by the RAGAS composite."""
    df = pd.read_csv(RUN / f'ragas_results_{ds}.csv')
    axes = df['variant'].str.extract(r'^(N\d+)_(E\d+[a-z]?)_(T\d+[a-z]?)$')
    axes.columns = AXES
    df = pd.concat([df.drop(columns=[c for c in AXES if c in df.columns]), axes], axis=1)

    split = pd.read_csv(RUN / f'question_split_{ds}.csv').set_index('question_id')['split']
    df = df[df['question_id'].isin(split.index)].copy()
    return _drop_duplicate_edges(
        _pool_means(df, df['question_id'].map(split), 'composite'), ds)


TASKS = {
    'node_classification': node_classification_variants,
    'graphrag': graphrag_variants,
}


# ── trees ─────────────────────────────────────────────────────────────────────

def fit(variants: pd.DataFrame, target: str) -> dict:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = enc.fit_transform(variants[AXES])
    y = variants[target].to_numpy(float)
    tree = DecisionTreeRegressor(**TREE_KWARGS).fit(X, y)
    return {'tree': tree, 'enc': enc, 'r2': float(tree.score(X, y)), 'n': len(variants)}


def apply_to(model: dict, variants: pd.DataFrame) -> np.ndarray:
    """Levels the source tree never saw become all-zero columns, not errors."""
    return model['tree'].predict(model['enc'].transform(variants[AXES]))


def _size(t, node: int) -> int:
    if t.feature[node] == -2:
        return 1
    return 1 + _size(t, t.children_left[node]) + _size(t, t.children_right[node])


def _edit(ta, tb, a: int = 0, b: int = 0) -> int:
    """Ordered top-down edit distance between two binary trees.

    Same position, different split feature -> 1 relabel.
    Leaf against a subtree -> insert or delete that subtree.
    """
    a_leaf, b_leaf = ta.feature[a] == -2, tb.feature[b] == -2
    if a_leaf and b_leaf:
        return 0
    if a_leaf:
        return _size(tb, b) - 1
    if b_leaf:
        return _size(ta, a) - 1
    cost = int(ta.feature[a] != tb.feature[b])
    return (cost
            + _edit(ta, tb, ta.children_left[a], tb.children_left[b])
            + _edit(ta, tb, ta.children_right[a], tb.children_right[b]))


def ted(model_a: dict, model_b: dict) -> float:
    ta, tb = model_a['tree'].tree_, model_b['tree'].tree_
    total = ta.node_count + tb.node_count
    return _edit(ta, tb) / total if total else np.nan


# ── the three measurements ────────────────────────────────────────────────────

def run_task(task: str, loader) -> dict:
    variants = {ds: loader(ds) for ds in DATASETS}
    train_tree = {ds: fit(variants[ds], 'train_mean') for ds in DATASETS}
    test_tree = {ds: fit(variants[ds], 'test_mean') for ds in DATASETS}

    rho = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
    pval = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
    ted_m = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)

    for a in DATASETS:
        for b in DATASETS:
            pred = apply_to(train_tree[a], variants[b])
            r, p = spearmanr(pred, variants[b]['test_mean'].to_numpy(float))
            rho.loc[a, b], pval.loc[a, b] = r, p
            # Diagonal: train-pool tree vs test-pool tree, same dataset.
            # Off-diagonal: the two datasets' train-pool trees.
            ted_m.loc[a, b] = ted(train_tree[a],
                                  test_tree[a] if a == b else train_tree[b])

    summary = pd.DataFrame([{
        'dataset': ds, 'n_variants': variants[ds].shape[0],
        'rho': rho.loc[ds, ds], 'p': pval.loc[ds, ds],
        'ted_within': ted_m.loc[ds, ds],
        'ted_cross': ted_m.loc[ds, [d for d in DATASETS if d != ds]].mean(),
        'train_r2': train_tree[ds]['r2'], 'test_r2': test_tree[ds]['r2'],
    } for ds in DATASETS])

    return {'variants': variants, 'rho': rho, 'p': pval, 'ted': ted_m,
            'summary': summary}


def random_ted(variants: pd.DataFrame) -> tuple[float, float]:
    """TED between two trees fit on independent noise over the same variants.

    Holds variant count and feature cardinality fixed, so it isolates how much
    structural agreement the encoding alone produces with no signal present.
    """
    vals = []
    for s in range(N_RANDOM_SEEDS):
        rng = np.random.RandomState(1000 + s)
        v = variants.copy()
        v['a'] = rng.normal(size=len(v))
        v['b'] = rng.normal(size=len(v))
        vals.append(ted(fit(v, 'a'), fit(v, 'b')))
    return float(np.mean(vals)), float(np.std(vals))


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def _cell(r: float, p: float, diag: bool, alpha: float) -> str:
    s = f'{r:+.3f}' + ('*' if p < alpha else '\\phantom{*}')
    return f'\\textbf{{{s}}}' if diag else s


def latex(results: dict, alpha_raw: float = 0.05) -> str:
    n_cells = len(DATASETS) ** 2
    alpha = alpha_raw / n_cells
    L = []
    L.append('\\begin{table}[htbp]')
    L.append('\\centering')
    L.append('\\renewcommand{\\arraystretch}{1.25}')
    L.append('\\small')
    L.append('\\setlength{\\tabcolsep}{3.7pt}')
    L.append('')
    L.append('\\caption{Decision-tree consistency, computed separately for each task '
             'and each dataset. \\textit{Left:} Spearman $\\rho$ between the ranking '
             'predicted by a tree fit on the train pool and the actual ranking on the '
             'held-out test pool, with rows the dataset the tree was fit on and columns '
             'the dataset it was applied to. The diagonal is held-out validation within '
             'a dataset. $^{*}$ marks $p<0.05$ after Bonferroni correction over the '
             f'{n_cells} cells of that panel. \\textit{{Right:}} normalized tree edit '
             'distance between the train-pool tree and the test-pool tree of the same '
             'dataset (Within), and the mean distance to the four other datasets\' '
             'trees (Cross). Lower means structurally more similar. $n$ is the number '
             'of variants surviving the zero-score filter.}')
    L.append('\\label{tab:dt_consistency}')
    L.append('')

    for task, heading in [('node_classification', 'Node Classification'),
                          ('graphrag', 'GraphRAG Retrieval')]:
        res = results[task]
        rho, p, ted_m = res['rho'], res['p'], res['ted']
        summ = res['summary'].set_index('dataset')

        L.append(f'\\textbf{{{heading}}}')
        L.append('')
        L.append('\\begin{tabular}[t]{lccccc}')
        L.append('\\toprule')
        L.append('$\\rho$ & ' + ' & '.join(SHORT[d] for d in DATASETS) + ' \\\\')
        L.append('\\midrule')
        for a in DATASETS:
            cells = [_cell(rho.loc[a, b], p.loc[a, b], a == b, alpha) for b in DATASETS]
            L.append(f'{SHORT[a]:<6}& ' + ' & '.join(cells) + ' \\\\')
        L.append('\\bottomrule')
        L.append('\\end{tabular}%')
        L.append('\\hspace{1.5em}')
        L.append('\\begin{tabular}[t]{lrcc}')
        L.append('\\toprule')
        L.append('Dataset & $n$ & Within & Cross \\\\')
        L.append('\\midrule')
        for d in DATASETS:
            L.append(f'{LABEL[d]:<12}& {int(summ.loc[d, "n_variants"])} & '
                     f'\\textbf{{{summ.loc[d, "ted_within"]:.3f}}} & '
                     f'{summ.loc[d, "ted_cross"]:.3f} \\\\')
        L.append('\\bottomrule')
        L.append('\\end{tabular}')
        L.append('')
        if task == 'node_classification':
            L.append('\\vspace{0.5em}')
            L.append('')

    L.append('\\end{table}')
    return '\n'.join(L)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None,
                    help='Defaults to analysis/dt_consistency for the '
                         'control-excluded run and analysis/'
                         'dt_consistency_with_control when --include_control '
                         'is set, so the two can never overwrite each other.')
    ap.add_argument('--include_control', action='store_true',
                    help='Keep the no-text control in the node-classification '
                         'variant set so counts match the cost table. Inflates rho.')
    ap.add_argument('--dedupe_edges', action='store_true',
                    help='Collapse edge rules that build identical graphs '
                         '(see code/detect_duplicate_edge_rules.py). Writes to '
                         'a _dedup suffixed directory.')
    args = ap.parse_args()
    global INCLUDE_CONTROL, DEDUPE_EDGES
    INCLUDE_CONTROL = args.include_control
    DEDUPE_EDGES = args.dedupe_edges
    if args.out is None:
        name = ('dt_consistency_with_control' if INCLUDE_CONTROL
                else 'dt_consistency')
        args.out = str(RUN / 'analysis' / (name + ('_dedup' if DEDUPE_EDGES else '')))
    print(f'no-text control: {"INCLUDED" if INCLUDE_CONTROL else "excluded"}')
    print(f'duplicate edge rules: {"COLLAPSED" if DEDUPE_EDGES else "kept"}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    rows = []
    for task, loader in TASKS.items():
        print(f'\n{"="*70}\n{task}\n{"="*70}')
        res = run_task(task, loader)
        results[task] = res

        for name, frame in [('rho', res['rho']), ('p', res['p']), ('ted', res['ted'])]:
            frame.to_csv(out / f'{task}_{name}.csv')
        res['summary'].to_csv(out / f'{task}_summary.csv', index=False)

        print(res['summary'].to_string(index=False,
              float_format=lambda v: f'{v:.4f}'))
        print('\n  rho matrix (row = tree source, col = target):')
        print(res['rho'].rename(index=LABEL, columns=LABEL).round(3).to_string())
        print('\n  TED matrix (diagonal = train tree vs test tree):')
        print(res['ted'].rename(index=LABEL, columns=LABEL).round(3).to_string())

        for ds in DATASETS:
            m, s = random_ted(res['variants'][ds])
            w = res['summary'].set_index('dataset').loc[ds, 'ted_within']
            rows.append({'task': task, 'dataset': ds,
                         'n_variants': res['variants'][ds].shape[0],
                         'ted_within': round(w, 4),
                         'random_mean': round(m, 4), 'random_std': round(s, 4),
                         'z': round((w - m) / s, 2) if s else np.nan})

    base = pd.DataFrame(rows)
    base.to_csv(out / 'ted_random_baseline.csv', index=False)
    print(f'\n{"="*70}\nrandom-score TED baseline ({N_RANDOM_SEEDS} seeds)\n{"="*70}')
    print(base.to_string(index=False))

    tex = latex(results)
    (out / 'table3_dt_consistency.tex').write_text(tex + '\n')
    print(f'\nwrote {out}/table3_dt_consistency.tex')


if __name__ == '__main__':
    main()
