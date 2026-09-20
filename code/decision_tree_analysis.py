#!/usr/bin/env python3
"""
SUPERSEDED BY code/dt_consistency.py -- do not use for new work.
It fits the same published quantity with min_samples_leaf=3, which
makes one-hot edge splits unreachable once the no-text control is
dropped; exposes only three of the five datasets in its CLI; and
reads output/construction_performance_table_*.csv, a path that no
longer exists. Retained because the files under
output/run_final/analysis/superseded/ were produced by it.

Fits a decision tree per dataset (plus a combined tree) predicting variant
mean score from the one-hot-encoded (Task, Node, Edge, Text) construction
axes, and renders each as an annotated PNG. Also exposes a programmatic API
(fit_tree / compare_trees / predict_and_validate) that run_analysis.py uses
directly instead of this file's own CLI.

Reads:
  - output/construction_performance_table_{dataset}.csv
    (dataset in {history, amazon, arxiv} for the CLI's --dataset flag —
    see docs/AUDIT_yaml_hardcoding.md finding #1: electronics/toys are not
    valid --dataset choices in this file's CLI, unlike the programmatic API)

Writes:
  - output/decision_trees/decision_tree_{scenario}.png

Usage:
    python3 code/decision_tree_analysis.py --dataset history
    python3 code/decision_tree_analysis.py --dataset amazon
    python3 code/decision_tree_analysis.py --dataset arxiv
    python3 code/decision_tree_analysis.py --dataset combined
    python3 code/decision_tree_analysis.py --dataset all
    python3 code/decision_tree_analysis.py --dataset all --depth 4 --open
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATA_DIR   = Path('output')
TREE_DIR   = Path('output/decision_trees')
SCORE_COL  = 'normalized_score'
GROUP_BASE = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']

# ── label maps ────────────────────────────────────────────────────────────────

LABEL_MAP = {
    'M1': 'Node Categ.',       'M2': 'Node Scalar',
    'M3': 'Edge Categ.',       'M4': 'Edge Scalar',
    'M5': 'Global Categ.',     'M6': 'Global Scalar',
    'N7': 'Primary Entity',    'N8': 'Secondary Entity',   'N9': 'Aggregate Entity',
    'E10a': 'Categ. GT',       'E10b': 'Co-partic. (wt.)',
    'E10c': 'Binary Partic.',  'E11a': 'Semantic Sim.',
    'E11b': 'Structural Sim.', 'E11c': 'Functional Sim.',
    'T12a': 'Contextual',      'T12b': 'Standard',         'T12e': 'Baseline',
    'amazon': 'Amazon',        'arxiv': 'ArXiv',           'history': 'History',
}

# Full-phrase labels for the PNG node boxes
FULL_LABEL = {
    'M1': 'Node Level\nCategorical',    'M2': 'Node Level\nScalar',
    'M3': 'Edge Level\nCategorical',    'M4': 'Edge Level\nScalar',
    'M5': 'Global\nCategorical',        'M6': 'Global\nScalar',
    'N7': 'Primary Entity',             'N8': 'Secondary Entity',
    'N9': 'Aggregate Entity',
    'E10a': 'Categorical GT',           'E10b': 'Weighted\nCo-participation',
    'E10c': 'Binary Participation',     'E11a': 'Semantic Similarity',
    'E11b': 'Structural Similarity',    'E11c': 'Functional Similarity',
    'T12a': 'Contextual\n(Title + Body)', 'T12b': 'Standard\n(Title Only)',
    'T12e': 'Baseline\n(Zero Embeds)',
    'amazon': 'Amazon',  'arxiv': 'ArXiv',  'history': 'History',
}

AXIS_PREFIX = {
    'Task_Idx': 'TASK',   'Node_Idx': 'NODE',
    'Edge_Idx': 'EDGE',   'Text_Idx': 'TEXT',
    'dataset':  'DATASET',
}

FEATURE_DISPLAY = {
    'Task_Idx': 'Task Type (M)',   'Node_Idx': 'Node Type (N)',
    'Edge_Idx': 'Edge Type (E)',   'Text_Idx': 'Text Fidelity (T)',
    'dataset':  'Dataset',
}


# ── color helpers ─────────────────────────────────────────────────────────────

def _internal_color(val: float):
    t = max(0.0, min(1.0, val / 100.0))
    return plt.colormaps['YlOrBr'](0.15 + t * 0.72)

def _leaf_color(val: float):
    t = max(0.0, min(1.0, val / 100.0))
    g = 0.08 + t * 0.87
    return (g, g, g, 1.0)

def _leaf_text_color(val: float) -> str:
    return 'white' if val / 100.0 < 0.55 else '#1a1a1a'

def _internal_text_color(val: float) -> str:
    return 'white' if val / 100.0 > 0.72 else '#1a1a1a'


# ── encoding ──────────────────────────────────────────────────────────────────

def build_ohe(variant_df: pd.DataFrame, feature_cols: list):
    """
    One-hot encode feature_cols so every tree split is an exact equality check.
    Returns (X, ohe_feature_names, ohe_to_axis_val).

    ohe_to_axis_val maps each OHE column name → (original_col, category_value)
    e.g. 'Task_Idx_M1' → ('Task_Idx', 'M1')
    """
    try:
        ohe = OneHotEncoder(sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(sparse=False)

    X = ohe.fit_transform(variant_df[feature_cols])
    ohe_names = list(ohe.get_feature_names_out(feature_cols))

    # Build reverse lookup: ohe column name → (axis_col, cat_value)
    ohe_to_axis_val = {}
    for i, col in enumerate(feature_cols):
        for cat in ohe.categories_[i]:
            ohe_to_axis_val[f'{col}_{cat}'] = (col, str(cat))

    return X, ohe_names, ohe_to_axis_val


# ── layout ────────────────────────────────────────────────────────────────────

def _leaf_positions(tree_model, node: int = 0, counter: list = None) -> dict:
    if counter is None:
        counter = [0]
    t = tree_model.tree_
    if t.feature[node] == -2:
        pos = {node: float(counter[0])}
        counter[0] += 1
        return pos
    left  = _leaf_positions(tree_model, t.children_left[node],  counter)
    right = _leaf_positions(tree_model, t.children_right[node], counter)
    lx = left[t.children_left[node]]
    rx = right[t.children_right[node]]
    pos = {node: (lx + rx) / 2.0}
    pos.update(left)
    pos.update(right)
    return pos


# ── custom tree plotter ───────────────────────────────────────────────────────

def plot_tree_categorical(
    tree_model,
    ohe_names: list,
    ohe_to_axis_val: dict,
    title: str = '',
    out_path=None,
    open_fig: bool = False,
):
    """
    Draw the decision tree. Every internal node shows:
        AXIS: Full Description     ← the single category being tested
        N=42,  Val=74%
    Left branch = NO (feature == 0), Right branch = YES (feature == 1).
    Leaf nodes are grayscale-colored by score.
    """
    t        = tree_model.tree_
    n_leaves = tree_model.get_n_leaves()
    depth    = tree_model.get_depth()
    node_x   = _leaf_positions(tree_model)

    # Relative colour scale: normalise to the tree's actual min/max
    _all_vals = [float(t.value[node][0][0]) for node in range(t.node_count)]
    _vmin, _vmax = min(_all_vals), max(_all_vals)
    _vrange = _vmax - _vmin if _vmax > _vmin else 1.0

    def _norm(v: float) -> float:
        return max(0.0, min(1.0, (v - _vmin) / _vrange))

    def _lc(v):   # leaf fill
        g = 0.08 + _norm(v) * 0.87
        return (g, g, g, 1.0)

    def _ltc(v):  # leaf text
        return 'white' if _norm(v) < 0.55 else '#1a1a1a'

    def _ic(v):   # internal fill
        return plt.colormaps['YlOrBr'](0.15 + _norm(v) * 0.72)

    def _itc(v):  # internal text
        return 'white' if _norm(v) > 0.72 else '#1a1a1a'

    FIG_W = max(40, 9.0 * n_leaves)
    FIG_H = max(20, 8.0 * (depth + 1))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    MARGIN = 0.7
    ax.set_xlim(-MARGIN, n_leaves - 1 + MARGIN)
    ax.set_ylim(-depth - MARGIN, 0.6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def draw(node: int, d: int):
        x   = node_x[node]
        y   = float(-d)
        n   = int(t.n_node_samples[node])
        val = float(t.value[node][0][0])

        if t.feature[node] == -2:
            # ── LEAF ─────────────────────────────────────────────────
            ax.text(
                x, y, f'{val:.0f}%',
                ha='center', va='center',
                fontsize=30, fontweight='bold',
                color=_ltc(val),
                bbox=dict(
                    boxstyle='round,pad=1.1',
                    facecolor=_lc(val),
                    edgecolor='#888888', linewidth=2.8,
                ),
                zorder=3,
            )
        else:
            # ── INTERNAL NODE ─────────────────────────────────────────
            feat_name          = ohe_names[t.feature[node]]
            axis_col, cat_val  = ohe_to_axis_val[feat_name]
            prefix             = AXIS_PREFIX.get(axis_col, axis_col.upper())
            desc               = FULL_LABEL.get(cat_val, cat_val)
            label              = f'{prefix}: {desc}\nN={n},  Val={val:.0f}%'

            ax.text(
                x, y, label,
                ha='center', va='center',
                fontsize=28, fontweight='bold',
                color=_itc(val),
                linespacing=1.6,
                bbox=dict(
                    boxstyle='round,pad=2.0',
                    facecolor=_ic(val),
                    edgecolor='#555555', linewidth=4.0,
                ),
                zorder=3,
            )

            # children: left = NO (feature==0), right = YES (feature==1)
            lc = t.children_left[node]   # NO
            rc = t.children_right[node]  # YES
            lx, ly = node_x[lc], float(-(d + 1))
            rx, ry = node_x[rc], float(-(d + 1))

            for cx, cy in [(lx, ly), (rx, ry)]:
                ax.annotate(
                    '', xy=(cx, cy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='#444444',
                                    lw=1.5, mutation_scale=14),
                    zorder=1,
                )

            def edge_label(tx, ty, text, ha):
                ax.text(tx, ty, text, ha=ha, va='center',
                        fontsize=24, fontstyle='italic', color='#333333',
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.85, pad=4.0),
                        zorder=4)

            edge_label(x + (lx - x) * 0.35 - 0.07,
                       y + (ly - y) * 0.35, 'No',  'right')
            edge_label(x + (rx - x) * 0.35 + 0.07,
                       y + (ry - y) * 0.35, 'Yes', 'left')

            draw(lc, d + 1)
            draw(rc, d + 1)

    draw(0, 0)

    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

    if out_path:
        fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
        print(f'\n  Saved → {out_path}')
    if open_fig:
        subprocess.run(['open', str(out_path)], check=False)
    plt.close(fig)


# ── terminal text decoder ─────────────────────────────────────────────────────

def decode_tree_text(tree_model, ohe_names: list, ohe_to_axis_val: dict) -> str:
    t = tree_model.tree_
    lines = []

    def recurse(node: int, depth: int):
        pad = '    ' * depth
        if t.feature[node] == -2:
            val = t.value[node][0][0]
            n   = int(t.n_node_samples[node])
            lines.append(f'{pad}→ score: {val:.1f}  (n={n})')
        else:
            feat_name         = ohe_names[t.feature[node]]
            axis_col, cat_val = ohe_to_axis_val[feat_name]
            disp  = FEATURE_DISPLAY.get(axis_col, axis_col)
            short = LABEL_MAP.get(cat_val, cat_val)
            n     = int(t.n_node_samples[node])
            lines.append(f'{pad}[{disp} == {short}?]  n={n}')
            lines.append(f'{pad}├── No')
            recurse(t.children_left[node],  depth + 1)
            lines.append(f'{pad}└── Yes')
            recurse(t.children_right[node], depth + 1)

    recurse(0, 0)
    return '\n'.join(lines)


# ── data helpers ──────────────────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    path = DATA_DIR / f'construction_performance_table_{name}.csv'
    df = pd.read_csv(path)
    df['dataset'] = name
    return df


def aggregate_variants(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    valid = df.dropna(subset=[SCORE_COL])

    def agg(split: str) -> pd.DataFrame:
        return (
            valid[valid['run_split'] == split]
            .groupby(group_cols)
            .agg(
                mean_score=(SCORE_COL, 'mean'),
                std_score=(SCORE_COL, 'std'),
                n_samples=(SCORE_COL, 'count'),
            )
            .reset_index()
        )

    train_v = agg('train')
    test_v  = agg('test')
    return train_v.merge(
        test_v[group_cols + ['mean_score', 'n_samples']],
        on=group_cols, suffixes=('_train', '_test'),
    )


# ── main routine per scenario ─────────────────────────────────────────────────

def run_tree(scenario: str, df: pd.DataFrame, max_depth: int, open_fig: bool) -> dict:
    is_combined = (scenario == 'combined')
    # For combined: aggregate by (dataset, M, N, E, T) so samples aren't
    # conflated across datasets, but train the tree on M/N/E/T only —
    # dataset is not a feature because the goal is general construction insight.
    group_cols   = (['dataset'] + GROUP_BASE) if is_combined else GROUP_BASE[:]
    feature_cols = GROUP_BASE[:]

    print(f'\n{"="*68}')
    print(f'  {scenario.upper()}')
    print(f'{"="*68}')

    variant_df = aggregate_variants(df, group_cols)
    n_var = len(variant_df)
    print(f'  {n_var} variants  |  '
          f'train score {variant_df["mean_score_train"].min():.1f}–'
          f'{variant_df["mean_score_train"].max():.1f}')

    X, ohe_names, ohe_to_axis_val = build_ohe(variant_df, feature_cols)
    y_train = variant_df['mean_score_train'].values
    y_test  = variant_df['mean_score_test'].values

    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=3, random_state=42)
    tree.fit(X, y_train)

    r2_train = tree.score(X, y_train)
    r2_test  = tree.score(X, y_test)
    print(f'  R²  train={r2_train:.3f}   test={r2_test:.3f}')

    # feature importances (aggregate back to original axes)
    imp_series = pd.Series(tree.feature_importances_, index=ohe_names)
    axis_imp   = {}
    for ohe_name, imp in imp_series.items():
        axis_col, _ = ohe_to_axis_val[ohe_name]
        axis_imp[axis_col] = axis_imp.get(axis_col, 0.0) + imp
    axis_imp = dict(sorted(axis_imp.items(), key=lambda x: x[1], reverse=True))

    print('\n  Feature importances (by axis):')
    for feat, imp in axis_imp.items():
        bar = '█' * int(imp * 32)
        print(f'    {FEATURE_DISPLAY.get(feat, feat):<22} {imp:.4f}  {bar}')

    print('\n  Decision tree (decoded):')
    txt = decode_tree_text(tree, ohe_names, ohe_to_axis_val)
    for line in txt.splitlines():
        print('  ' + line)

    print('\n  Mean score by axis  (train | test):')
    for col in GROUP_BASE:
        if col not in variant_df.columns:
            continue
        bd = (
            variant_df.groupby(col)[['mean_score_train', 'mean_score_test']]
            .mean().round(1)
            .sort_values('mean_score_train', ascending=False)
        )
        print(f'\n    {FEATURE_DISPLAY.get(col, col)}:')
        for idx_val, row in bd.iterrows():
            short = LABEL_MAP.get(idx_val, idx_val)
            print(f'      {idx_val}  {short:<26}  '
                  f'train={row["mean_score_train"]:5.1f}  '
                  f'test={row["mean_score_test"]:5.1f}')

    TREE_DIR.mkdir(parents=True, exist_ok=True)
    out_png = TREE_DIR / f'decision_tree_{scenario}.png'

    plot_tree_categorical(
        tree_model=tree,
        ohe_names=ohe_names,
        ohe_to_axis_val=ohe_to_axis_val,
        title=(
            f'TAG Construction Decision Tree — {scenario}\n'
            f'depth={max_depth}   train R²={r2_train:.3f}   test R²={r2_test:.3f}'
            f'   N={n_var} variants'
        ),
        out_path=out_png,
        open_fig=open_fig,
    )

    return {
        'scenario':   scenario,
        'n_variants': n_var,
        'r2_train':   round(r2_train, 3),
        'r2_test':    round(r2_test,  3),
        'axis_importances': {FEATURE_DISPLAY.get(k, k): round(v, 4)
                             for k, v in axis_imp.items()},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train and visualize TAG construction decision trees.'
    )
    parser.add_argument(
        '--dataset',
        choices=['history', 'amazon', 'arxiv', 'combined', 'all'],
        default='all',
    )
    parser.add_argument('--depth', type=int, default=8,
                        help='Max tree depth (default: 8)')
    parser.add_argument('--open', dest='open_fig', action='store_true',
                        help='Open each PNG immediately after saving (macOS)')
    args = parser.parse_args()

    all_dfs: dict = {}
    for ds in ['history', 'amazon', 'arxiv']:
        p = DATA_DIR / f'construction_performance_table_{ds}.csv'
        if p.exists():
            all_dfs[ds] = load_dataset(ds)
        else:
            print(f'WARNING: {p} not found — {ds} will be skipped')

    scenarios = (
        ['history', 'amazon', 'arxiv', 'combined']
        if args.dataset == 'all'
        else [args.dataset]
    )

    results = []
    for scenario in scenarios:
        if scenario == 'combined':
            if len(all_dfs) < 2:
                print('Need ≥2 datasets for combined — skipping')
                continue
            df = pd.concat(all_dfs.values(), ignore_index=True)
        else:
            if scenario not in all_dfs:
                print(f'  SKIP {scenario} (CSV not found)')
                continue
            df = all_dfs[scenario]

        results.append(run_tree(scenario, df, max_depth=args.depth,
                                open_fig=args.open_fig))

    if len(results) > 1:
        print(f'\n{"="*68}')
        print('  SUMMARY')
        print(f'{"="*68}')
        summary = pd.DataFrame(results)[['scenario', 'n_variants', 'r2_train', 'r2_test']]
        print(summary.to_string(index=False))

    print(f'\nDone. PNGs saved in {TREE_DIR}/')


# ═══════════════════════════════════════════════════════════════════════════════
# Programmatic API — fit_tree / compare_trees / predict_and_validate
# (Tasks 1-3 from the analysis spec)
# ═══════════════════════════════════════════════════════════════════════════════

def build_variant_summary(
    df: pd.DataFrame,
    group_cols: list = None,
    score_col: str = None,
) -> pd.DataFrame:
    """
    Task 1 — Per-variant aggregation.

    For each (Task_Idx, Node_Idx, Edge_Idx, Text_Idx) variant compute per-split
    mean, std, and non-NaN count of `score_col` (defaults to normalized_score).
    NaN rows are excluded; they are NOT treated as 0.

    Returned columns:
        task_type, Task_Idx, Node_Idx, Edge_Idx, Text_Idx,
        train_mean, train_std, n_train_valid,
        test_mean,  test_std,  n_test_valid
    """
    if group_cols is None:
        group_cols = ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']
    if score_col is None:
        score_col = SCORE_COL

    extra = list(group_cols)
    if 'task_type' in df.columns and 'task_type' not in extra:
        extra = ['task_type'] + extra

    valid = df.dropna(subset=[score_col])

    def _agg(split: str) -> pd.DataFrame:
        return (
            valid[valid['run_split'] == split]
            .groupby(extra, sort=False)[score_col]
            .agg(**{
                f'{split}_mean':    'mean',
                f'{split}_std':     'std',
                f'n_{split}_valid': 'count',
            })
            .reset_index()
        )

    train_agg = _agg('train')
    test_agg  = _agg('test')
    merged    = train_agg.merge(test_agg, on=extra, how='outer')

    col_order = extra + [
        'train_mean', 'train_std', 'n_train_valid',
        'test_mean',  'test_std',  'n_test_valid',
    ]
    return merged[[c for c in col_order if c in merged.columns]]


def zero_inflation_diagnostic(
    df: pd.DataFrame,
    score_col: str = None,
) -> pd.DataFrame:
    """
    Task 2 — Zero-inflation diagnostic.

    Reports, per Task_Idx (M1-M6), what fraction of individual sample rows
    have normalized_score == 0.0 exactly (not NaN).  Rows are grouped by the
    Step-1 formula family:
        pseudo-R2  : M1, M3  (McFadden pseudo-R² → max(0,·))
        R2         : M2, M4  (standard R² → max(0,·))
        TVD-ratio  : M5      (TVD-relative improvement)
        gap-ratio  : M6      (median-gap-relative improvement)

    Returns one row per Task_Idx with columns:
        task_idx, task_type, step1_formula,
        n_total, n_zero, pct_zero, n_nan, n_degen
    """
    if score_col is None:
        score_col = SCORE_COL

    FAMILY = {
        'M1': 'pseudo-R2', 'M3': 'pseudo-R2',
        'M2': 'R2',        'M4': 'R2',
        'M5': 'TVD-ratio', 'M6': 'gap-ratio',
    }

    task_col = 'Task_Idx' if 'Task_Idx' in df.columns else 'task_type'
    type_col = 'task_type' if 'task_type' in df.columns else None

    rows = []
    for key, sub in df.groupby(task_col):
        n_total = len(sub)
        n_zero  = int((sub[score_col] == 0.0).sum())
        n_nan   = int(sub[score_col].isna().sum())
        n_degen = int(sub['degenerate'].sum()) if 'degenerate' in sub.columns else 0
        tt      = sub[type_col].iloc[0] if type_col else 'unknown'
        rows.append({
            'task_idx':     key,
            'task_type':    tt,
            'step1_formula': FAMILY.get(str(key), 'unknown'),
            'n_total':      n_total,
            'n_zero':       n_zero,
            'pct_zero':     round(100.0 * n_zero / n_total, 1) if n_total else float('nan'),
            'n_nan':        n_nan,
            'n_degen':      n_degen,
        })
    return pd.DataFrame(rows)


def fit_tree(
    variant_score_df: pd.DataFrame,
    target_col: str,
    features: list = None,
    max_depth: int = 8,
) -> dict:
    """
    Task 3 — Fit a DecisionTreeRegressor on one-hot-encoded construction axes.

    Parameters
    ----------
    variant_score_df : one row per variant; must contain feature columns and
        target_col.  Typically the output of build_variant_summary().
    target_col : score column to predict (e.g. 'train_mean' or 'test_mean').
    features : categorical feature columns.  Defaults to whichever of
        ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx'] exist in the frame.
    max_depth : maximum tree depth (default 8).

    Returns
    -------
    dict with keys:
        tree, ohe, ohe_names, ohe_to_axis_val, feature_cols,
        X, y, variant_df, target_col, r2
    """
    if features is None:
        features = [c for c in ['Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx']
                    if c in variant_score_df.columns]

    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    X = ohe.fit_transform(variant_score_df[features])
    ohe_names       = list(ohe.get_feature_names_out(features))
    ohe_to_axis_val = {
        f'{col}_{cat}': (col, str(cat))
        for i, col in enumerate(features)
        for cat in ohe.categories_[i]
    }

    y          = variant_score_df[target_col].values.astype(float)
    valid_mask = ~np.isnan(y)

    tree = DecisionTreeRegressor(
        criterion='squared_error', max_depth=max_depth,
        min_samples_leaf=3, random_state=42,
    )
    tree.fit(X[valid_mask], y[valid_mask])

    return {
        'tree':            tree,
        'ohe':             ohe,
        'ohe_names':       ohe_names,
        'ohe_to_axis_val': ohe_to_axis_val,
        'feature_cols':    features,
        'X':               X,
        'y':               y,
        'variant_df':      variant_score_df,
        'target_col':      target_col,
        'r2':              float(tree.score(X[valid_mask], y[valid_mask])),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _encode_for_tree(tree_result: dict, df: pd.DataFrame) -> np.ndarray:
    """Apply tree_result's fitted OHE to df.  Unknown categories → all-zeros."""
    return tree_result['ohe'].transform(df[tree_result['feature_cols']])


def _count_nodes(t, node: int) -> int:
    if t.feature[node] == -2:
        return 1
    return (1
            + _count_nodes(t, t.children_left[node])
            + _count_nodes(t, t.children_right[node]))


def _tree_edit_distance(tree_a, tree_b, na: int = 0, nb: int = 0) -> int:
    """
    Ordered top-down binary tree edit distance.

    Aligns trees by position (same root-to-node path) rather than computing the
    globally optimal Zhang-Shasha correspondence.  Cost model:
      - Different split feature at the same position → 1 (substitution)
      - One tree has stopped splitting (leaf vs. subtree) → subtree_size − 1
        (insert or delete the excess subtree)
    This approximation is exact when trees have identical structure; for trees
    of depth ≤ 8 with OHE binary splits it gives a useful structural distance
    without the O(n²m²) overhead of full Zhang-Shasha.
    """
    ta, tb    = tree_a.tree_, tree_b.tree_
    a_is_leaf = ta.feature[na] == -2
    b_is_leaf = tb.feature[nb] == -2

    if a_is_leaf and b_is_leaf:
        return 0
    if a_is_leaf:
        return _count_nodes(tb, nb) - 1
    if b_is_leaf:
        return _count_nodes(ta, na) - 1

    sub = 0 if ta.feature[na] == tb.feature[nb] else 1
    return (sub
            + _tree_edit_distance(tree_a, tree_b,
                                  ta.children_left[na],  tb.children_left[nb])
            + _tree_edit_distance(tree_a, tree_b,
                                  ta.children_right[na], tb.children_right[nb]))


def _assign_bands(scores: np.ndarray, task_types: np.ndarray = None) -> np.ndarray:
    """
    Assign 'high' / 'mid' / 'low' bands using tertile thresholds (33rd / 67th
    percentile) computed within each task_type group when supplied, else globally.
    This matches the methodology doc's percentile-based band definition.
    """
    bands = np.full(len(scores), 'mid', dtype=object)

    def _apply(mask, s):
        if mask.sum() < 3:
            return
        lo, hi = np.percentile(s[mask], [33.33, 66.67])
        bands[mask & (scores >= hi)] = 'high'
        bands[mask & (scores <= lo)] = 'low'

    if task_types is not None and len(task_types) == len(scores):
        for tt in np.unique(task_types):
            _apply(task_types == tt, scores)
    else:
        _apply(np.ones(len(scores), dtype=bool), scores)

    return bands


# ── public comparison functions ───────────────────────────────────────────────

def compare_trees(
    tree_result_a: dict,
    tree_result_b: dict,
    variant_score_df: pd.DataFrame,
) -> dict:
    """
    Task 3 — Compare two fitted trees on the same variant set.

    Both trees must have been fit with the same feature columns.
    tree_result_a's OHE is used for encoding (handle_unknown='ignore' makes
    this safe even when categories differ slightly between fits).

    Returns
    -------
    dict:
        tree_edit_distance : int   — ordered top-down TED
        spearman_rho       : float — Spearman ρ between tree_a and tree_b predictions
        spearman_p         : float — two-sided p-value
        band_accuracy      : float — fraction of variants where band (H/M/L) agrees
    """
    from scipy.stats import spearmanr

    X      = _encode_for_tree(tree_result_a, variant_score_df)
    pred_a = tree_result_a['tree'].predict(X)
    pred_b = tree_result_b['tree'].predict(X)

    tt = (variant_score_df['task_type'].values
          if 'task_type' in variant_score_df.columns else None)
    bands_a = _assign_bands(pred_a, tt)
    bands_b = _assign_bands(pred_b, tt)

    rho, p = spearmanr(pred_a, pred_b)
    ted    = _tree_edit_distance(tree_result_a['tree'], tree_result_b['tree'])

    return {
        'tree_edit_distance': int(ted),
        'spearman_rho':       float(rho),
        'spearman_p':         float(p),
        'band_accuracy':      float((bands_a == bands_b).mean()),
    }


def predict_and_validate(
    tree_result_a: dict,
    score_df_b: pd.DataFrame,
    target_col: str = None,
) -> dict:
    """
    Task 3 — Apply a tree fit on one data slice to predict scores on a different slice.

    Parameters
    ----------
    tree_result_a : dict from fit_tree() — contains tree, ohe, feature_cols.
    score_df_b    : DataFrame for the target slice; must contain the same feature
                    columns and target_col.  Can be a different split, dataset, or pool.
    target_col    : column in score_df_b to validate against.
                    Defaults to tree_result_a['target_col'].

    Returns
    -------
    dict:
        spearman_rho  : float — Spearman ρ (predicted vs actual)
        spearman_p    : float — two-sided p-value
        band_accuracy : float — fraction of variants where predicted band matches
                                actual band (tertile, per task_type)
        predictions   : np.ndarray — raw predicted scores for all rows in score_df_b
        actual        : np.ndarray — raw actual scores from target_col
    """
    from scipy.stats import spearmanr

    if target_col is None:
        target_col = tree_result_a['target_col']

    X_b    = _encode_for_tree(tree_result_a, score_df_b)
    pred   = tree_result_a['tree'].predict(X_b)
    actual = score_df_b[target_col].values.astype(float)

    valid        = ~np.isnan(actual)
    pred_v       = pred[valid]
    actual_v     = actual[valid]
    tt           = (score_df_b['task_type'].values[valid]
                    if 'task_type' in score_df_b.columns else None)
    bands_pred   = _assign_bands(pred_v,   tt)
    bands_actual = _assign_bands(actual_v, tt)

    rho, p = spearmanr(pred_v, actual_v)

    return {
        'spearman_rho':  float(rho),
        'spearman_p':    float(p),
        'band_accuracy': float((bands_pred == bands_actual).mean()),
        'predictions':   pred,
        'actual':        actual,
    }


if __name__ == '__main__':
    main()
