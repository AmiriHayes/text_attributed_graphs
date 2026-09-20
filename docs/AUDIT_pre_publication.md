# Pre-Publication Implementation Audit

Audit of M/N/E/T construction axes across all five datasets (ArXiv, Amazon,
History, Electronics, Toys), triggered by the E11b finding (k=50 default
degenerating under centrality-tie clustering, masquerading as a "structural
similarity edges are bad" finding). This file consolidates the original
audit and the four code fixes applied afterward.

Confidence key: **VERIFIED** = empirically checked against live data this
session. **CODE-REVIEWED** = assessed by reading the implementation, not
independently re-run at full scale.

---

## CRITICAL findings

### C1 — E10a ≡ E11c (identical implementation)

**Finding (VERIFIED via code inspection):** `edge_factory.py` dispatched
E11c to the exact same function as E10a (`_build_categorical_gt`), with
identical arguments. Every E11c graph was byte-identical to its E10a
counterpart, not merely similar.

**Scope, corrected after reading all 5 `{dataset}_variants.yaml` files:**
E10a/E11c are excluded from M1 for **N7 in all 5 datasets** and **N9 in
all 5 datasets** (both documented as C4 label leaks / tautological —
`categorical_label` is a deterministic function of `aggregate_id` for
every dataset, and N9 nodes *are* aggregate_id). They are further excluded
entirely for **ArXiv and History** even at N8 (mode(aggregate_id) over an
author's papers is still considered a C4 leak there). The bug therefore
only actually affected live M1 variants at **(M1, N8)** in **Amazon,
Electronics, Toys** — 3 variants per dataset (one per text fidelity), 9
total. It has zero impact on any published ArXiv or History M1 result.

**Fix applied:** Considered implementing a genuinely distinct hierarchical
E11c (soft grouping one level above `categorical_label`) and rejected it
as intractable without new data engineering:
- ArXiv/History: `categorical_label` already *is* the top-level/root
  category per their own `_variants.yaml` documentation — no coarser level
  exists in the current per-sample schema.
- Amazon-family: a coarser level exists in the raw meta `categories` list
  (index 0/1 vs. the index 2 used for `l3_cat`), but extracting it
  requires re-streaming `m1_meta_url` into a new cache — a data-fetch
  operation, not a code fix, and out of scope for this pass.

Per the audit's own permitted fallback, **E11c is deprecated**, not left
as a silent alias: `edge_factory.py` now raises `NotImplementedError` with
a full explanation if E11c is requested. The 9 `(M1, N8, E11c)` lines were
removed from `amazon_variants.yaml`, `electronics_variants.yaml`,
`toys_variants.yaml` (the only files where they were live).

**Rerun needed:** None. This is a labeling/taxonomy correction — the E10a
numbers already in every construction_performance_table are correct and
unaffected; only the redundant E11c rows/labels need dropping from any
downstream table or figure that lists them as a 6th distinct edge type.

---

### C2 — N8 hub nodes are single-item for the large majority

**Finding (VERIFIED, sample_00, all 4 datasets with N8):**

| dataset | N8 count | mean N7-per-N8 | frac exactly 1 |
|---|---|---|---|
| ArXiv | 2,919 | 1.02 | 98.0% |
| Amazon | 881 | 1.14 | 90.7% |
| Electronics | 854 | 1.17 | 89.2% |
| Toys | 732 | 1.37 (std 1.99, max 45) | 86.7% |

For 87-98% of "hub" nodes, there is no aggregation happening — an N8 node
is functionally identical to its single associated N7 node. This does
**not** falsify the empirical N8-correlates-with-RAGAS-quality finding
(N8 graphs are still topologically distinct from N7 graphs via N8-N8
co-participation edges, independent of feature aggregation) — but it does
mean the *mechanistic explanation* ("hub nodes create richer semantic
clusters via aggregation") is unsupported for the large majority of nodes
as currently constructed.

**Fix applied:** Added a `minimum_hub_size` config parameter (default 1 =
current/unfiltered behavior) to `GenericDataManager.get_node_list`'s N8
branch. Filters N8 nodes to those with `>= minimum_hub_size` associated N7
nodes before the graph is built.

**Threshold sweep (Amazon, VERIFIED, both scales):**

*1000-row training-sample scale (used for raw_gnn / construction_performance_table):*

| min_hub_size | N8 count | mean N7/N8 | frac singleton | E10b edges |
|---|---|---|---|---|
| 1 (current) | 881 | 1.14 | 90.7% | 22 |
| 2 | 82 | 2.45 | 0.0% | 1 |
| 3 | 19 | 3.95 | 0.0% | 0 |

*Pooled scale, 8,610 rows (used for GraphRAG/RAGAS graphs):*

| min_hub_size | N8 count | mean N7/N8 | frac singleton | E10b edges |
|---|---|---|---|---|
| 1 (current) | 5,519 | 1.56 | 76.6% | 0 |
| 2 | 1,290 | 3.40 | 0.0% | 0 |
| 3 | 545 | 5.31 | 0.0% | 0 |
| 4 | 296 | 7.24 | 0.0% | 0 |
| 5 | 186 | 9.16 | 0.0% | 0 |

**No single threshold satisfies both pipelines.** At GNN-training-sample
scale, `min_hub_size=2` already drops below the 100-node usable-graph
floor (82 nodes) — there is no viable filtered value at this scale for
Amazon. At the pooled GraphRAG-graph scale, `min_hub_size=2` or `3` both
comfortably clear the floor (1,290 / 545 nodes) with genuine aggregation
(mean 3.4-5.3 N7 per hub, 0% singletons).

**Decision:** `minimum_hub_size` left at **1** (default/unfiltered) in all
dataset yamls for now, since that is the only value compatible with the
existing GNN training pipeline at its current sample scale. The parameter
is implemented and verified working; choosing a non-default value requires
deciding whether to (a) only re-test the GraphRAG/RAGAS side at
`min_hub_size=2-3` (pooled scale, where it's viable) while leaving raw_gnn
untouched, or (b) redesign the GNN pipeline's sample scale for N8
specifically. Recommend (a) — see rerun plan.

**E10b (co-participation) edges collapse toward zero as the filter
tightens**, at both scales — filtering to genuine hubs removes almost all
remaining co-participation structure among them. This is a real, separate
side effect worth noting in any write-up of this fix.

---

### C3 — T12e silently equals T12b for N8 and N9 (not a zero vector)

**Finding (VERIFIED):** `get_embeddings()`'s N8 and N9 branches never
checked the `fidelity` argument — they always loaded the real embeddings
file regardless of what fidelity was requested. Direct confirmation
(pre-fix): `amazon N8: T12e == T12b? True` (mean abs value 0.029, not
remotely zero). N9 confirmed identical.

**Impact:** Every N8/N9/T12e row in every dataset's construction
performance table was silently a duplicate of the corresponding T12b row.
No text-fidelity ablation (T12a vs T12b vs T12e) has ever actually been
tested for N8 or N9, in any dataset, prior to this fix. Any claim of the
form "text fidelity matters less for hub/aggregate node types" (e.g. the
low `Text_Idx` importance seen for N8/N9 in this project's decision trees)
is uninterpretable until rerun, since T12e never differed from T12b in the
first place for those node types.

**Fix applied:** Added explicit `if fidelity == 'T12e': return
np.zeros(...)` blocks to both the N8 and N9 branches, mirroring the
existing N7 branch exactly.

**Verification (Amazon, sample_00, VERIFIED post-fix):**
```
N8: T12e all-zero=True   T12e==T12b=False
N9: T12e all-zero=True   T12e==T12b=False
```

**Rerun needed:** Every `(N8, *, T12e)` and `(N9, *, T12e)` M1/M3/M4/M5/M6
row across all 5 datasets. These rows in the existing
`construction_performance_table_{dataset}.csv` files are now known
incorrect and must be excluded from any text-fidelity claim until rerun.

---

### C4 — E11b k=50 default degenerates under centrality-tie clustering (prior finding, fix now applied)

**Recap:** centrality ties measured 37.5%-100% of nodes across datasets
checked, causing k-NN-on-centrality to degenerate into near-complete
graphs (100-650x denser than E10b/E11a on the same node set). No k value
fully escapes this — even k=1 (the sparsest possible) still produces
roughly one edge per node when ties are this common.

**Fix applied:** `_build_structural_similarity`'s default changed from
`k=50` to `k=5`. `k` is now threaded through as a configurable
`k_structural` kwarg in `EdgeFactory.build_edges` → `TAGConstructor`, and
documented per-dataset as `k_structural: 5` in each `{dataset}_dataset.yaml`
(the code default already covers it; the yaml entry is for discoverability).
Full docstring in `edge_factory.py` documents the degenerate condition and
the validation experiment.

**Validation experiment (ArXiv, N7, T12a, VERIFIED):**
- raw_gnn (GraphSAGE M1, n=10 samples): k=50 mean=55.79 (std 2.62) vs.
  k=5 mean=54.34 (std 3.42). Welch's t=-1.065, p=0.302 (not significant),
  Cohen's d=-0.476.
- RAGAS composite (n=20 questions): k=50=0.359 vs. k=5=0.639 (+0.28).
  Moves E11b from a catastrophic outlier to just below the bottom of the
  normal N7 range (0.687-0.719 for the other 4 edge types).

**Post-fix edge-count verification (Amazon, N7, sample_00, VERIFIED):**

| edge type | edges |
|---|---|
| E10b | 208 |
| E10c | 208 |
| E11a | 77 |
| **E11b (k=5, was k=50)** | **4,811** |

k=5 is a ~90% reduction from what k=50 would produce at this scale, but
E11b remains meaningfully denser than the other edge types (23-62x) — the
fix resolves the *degenerate/near-complete* failure mode, it does not make
E11b's density fully comparable to the ground-truth/similarity edge types.
This matches the ArXiv finding exactly (no k value closes that gap
entirely).

**Important caveat carried forward:** the validation experiment is one
dataset/node-type/text-fidelity combination (ArXiv, N7, T12a). Treat as a
validated pilot, not a five-dataset confirmation, until replicated — Amazon
N8 is the natural next check, given it showed the most severe centrality
tying observed (100%).

**Rerun needed:** Every `(*, E11b, *)` variant across all 5 datasets.

---

## MODERATE / MINOR (carried forward, not re-verified in this pass)

- **E11a threshold=0.75** (MODERATE): confirmed non-degenerate but
  apparently untuned — ArXiv/Amazon random-pair cosine similarity P99 is
  0.458/0.447, far below the 0.75 cutoff, meaning it selects only the
  extreme near-duplicate tail. Not fixed in this pass (task did not
  request it); worth a documentation note in the paper.
- **M5 TVD small-test-mask noise** (MODERATE, CODE-REVIEWED only):
  consistent with, and likely explains, the Electronics M5 reliability
  instability found earlier in this project at 500-row scale. Not
  addressed in this pass.
- Items not re-audited in this pass: E10a/E10b edge-density tables, M4
  zero-helpful-vote fraction, T12a/T12b length/null-rate tables. No
  numbers fabricated for these — flagged as open in the original audit.
