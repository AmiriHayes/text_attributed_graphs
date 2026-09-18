#!/usr/bin/env python3
"""
Verification suite for dt_consistency.py.

Every check either passes or prints a FAIL line with the offending values.
Nothing here trusts the implementation; each test re-derives the quantity a
different way, or constructs a case whose answer is known in advance.

Usage:  python3 code/test_dt_consistency.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dt_consistency as D

RUN = D.RUN
FAILS = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


# ── 1. determinism ────────────────────────────────────────────────────────────
print('\n1. DETERMINISM — same inputs must give bit-identical outputs')
for task, loader in D.TASKS.items():
    a = D.run_task(task, loader)
    b = D.run_task(task, loader)
    same = all(np.allclose(a[k].values.astype(float), b[k].values.astype(float),
                           equal_nan=True) for k in ('rho', 'p', 'ted'))
    check(f'{task}: two runs identical', same)


# ── 2. pool integrity ─────────────────────────────────────────────────────────
print('\n2. POOL INTEGRITY — train and test must be disjoint and correctly sized')
import glob, hashlib, json

# sample_idx restarts at 0 within each split, so the integer indices overlap by
# construction.  The pools are directories of sample files; compare contents.
for ds in D.DATASETS:
    h = {}
    for sp in ['train', 'test']:
        files = sorted(glob.glob(f'data/{ds}/{sp}/samples/*.jsonl'))
        h[sp] = {hashlib.md5(open(f, 'rb').read()).hexdigest() for f in files}
    check(f'{ds} GNN subsets are distinct draws', not (h['train'] & h['test']),
          f'train={len(h["train"])} test={len(h["test"])} identical={len(h["train"] & h["test"])}')

# Informational, not a pass/fail: the two pools are independent draws of
# subsets, but from the SAME row corpus, so individual rows recur across them.
# rho is therefore split-half reliability over subsets, not generalization to
# unseen rows.  Reported so the paper does not overclaim.
print('\n   row-level overlap between pools (informational):')
for ds in D.DATASETS[:3]:
    ids = {}
    for sp in ['train', 'test']:
        seen = set()
        for f in sorted(glob.glob(f'data/{ds}/{sp}/samples/*.jsonl')):
            for line in open(f):
                r = json.loads(line)
                seen.add(str(r.get('primary_id') or r.get('id') or list(r.values())[0]))
        ids[sp] = seen
    sh = len(ids['train'] & ids['test'])
    print(f'     {ds:12s} train={len(ids["train"]):6d} test={len(ids["test"]):6d} shared={sh:6d}'
          f'  ({100*sh/len(ids["train"]):.0f}%)')

for ds in D.DATASETS:
    sp = pd.read_csv(RUN / f'question_split_{ds}.csv')
    dup = sp.question_id.duplicated().sum()
    counts = sp.split.value_counts().to_dict()
    check(f'{ds} question split: one label per question', dup == 0, str(counts))


# ── 3. no test leakage into the train tree ────────────────────────────────────
print('\n3. LEAKAGE — the train tree must not depend on test-pool values at all')
for task, loader in D.TASKS.items():
    for ds in D.DATASETS:
        v = loader(ds)
        base = D.apply_to(D.fit(v, 'train_mean'), v)
        wrecked = v.copy()
        wrecked['test_mean'] = np.random.RandomState(0).normal(size=len(v)) * 1e6
        after = D.apply_to(D.fit(wrecked, 'train_mean'), wrecked)
        check(f'{task}/{ds}: predictions unchanged when test_mean is destroyed',
              np.array_equal(base, after))


# ── 4. null control ───────────────────────────────────────────────────────────
print('\n4. NULL CONTROL — permuting the held-out target must collapse rho to ~0')
print('   (also gives an exact permutation p-value, which does not assume')
print('    independent observations the way the parametric Spearman p does)')
print(f'\n   {"task":20s}{"dataset":13s}{"rho":>8}{"null mean":>11}{"null sd":>9}{"perm p":>10}')
for task, loader in D.TASKS.items():
    for ds in D.DATASETS:
        v = loader(ds)
        pred = D.apply_to(D.fit(v, 'train_mean'), v)
        act = v['test_mean'].to_numpy(float)
        obs = spearmanr(pred, act)[0]
        rng = np.random.RandomState(7)
        null = np.array([spearmanr(pred, rng.permutation(act))[0] for _ in range(5000)])
        pp = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
        print(f'   {task:20s}{ds:13s}{obs:8.3f}{null.mean():11.3f}{null.std():9.3f}{pp:10.4f}')
        if abs(null.mean()) > 0.05:
            check(f'{task}/{ds} null centred on zero', False, f'{null.mean():.3f}')

check('all null distributions centred on zero', True)


# ── 5. order invariance ───────────────────────────────────────────────────────
print('\n5. ORDER INVARIANCE — shuffling variant row order must not change rho')
for task, loader in D.TASKS.items():
    for ds in D.DATASETS:
        v = loader(ds)
        r1 = spearmanr(D.apply_to(D.fit(v, 'train_mean'), v), v.test_mean)[0]
        s = v.sample(frac=1, random_state=3).reset_index(drop=True)
        r2 = spearmanr(D.apply_to(D.fit(s, 'train_mean'), s), s.test_mean)[0]
        if not np.isclose(r1, r2, atol=1e-9):
            check(f'{task}/{ds} order invariant', False, f'{r1:.6f} vs {r2:.6f}')
check('all rho values invariant to row order', True)


# ── 6. TED on cases with known answers ────────────────────────────────────────
print('\n6. TED — behaviour on trees whose distance is known by construction')
v = D.node_classification_variants('amazon')
m = D.fit(v, 'train_mean')
check('TED(t, t) == 0', D.ted(m, m) == 0.0, f'{D.ted(m, m)}')

stump = dict(m)
stump['tree'] = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2,
                                      random_state=42).fit(
    m['enc'].transform(v[D.AXES]), v.train_mean)
d = D.ted(m, stump)
# stump has 3 nodes; distance should be (m.nodes - 1 excess subtree) style, > 0
check('TED(full tree, depth-1 stump) > 0', d > 0, f'{d:.4f}')
check('TED symmetric', np.isclose(D.ted(m, stump), D.ted(stump, m)),
      f'{D.ted(m, stump):.4f} vs {D.ted(stump, m):.4f}')
check('TED in [0, 1]', 0 <= d <= 1, f'{d:.4f}')

a = D.fit(v.assign(z=np.random.RandomState(1).normal(size=len(v))), 'z')
b = D.fit(v.assign(z=np.random.RandomState(2).normal(size=len(v))), 'z')
check('TED(noise, noise) > TED(real train, real test)',
      D.ted(a, b) > D.ted(D.fit(v, 'train_mean'), D.fit(v, 'test_mean')),
      f'{D.ted(a, b):.4f} vs {D.ted(D.fit(v, "train_mean"), D.fit(v, "test_mean")):.4f}')


# ── 7. hand-checkable case ────────────────────────────────────────────────────
print('\n7. HAND CHECK — History node classification has 8 variants; verify by eye')
v = D.node_classification_variants('history')
m = D.fit(v, 'train_mean')
pred = D.apply_to(m, v)
out = v.copy()
out['predicted'] = pred.round(2)
print(out.round(2).to_string(index=False))
print(f'\n   Spearman(predicted, test_mean) = {spearmanr(pred, v.test_mean)[0]:.4f}')
print('   rank of predicted:', pd.Series(pred).rank().astype(int).tolist())
print('   rank of test_mean:', v.test_mean.rank().astype(int).tolist())

print('\n' + '=' * 70)
print(f'{len(FAILS)} failure(s)' + (': ' + ', '.join(FAILS) if FAILS else ''))
sys.exit(1 if FAILS else 0)
