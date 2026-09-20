# YAML Configuration Hardcoding Audit

Scope: every `.py` file in `code/` (21 files), read in full. Goal: find places
where dataset-specific behavior is hardcoded in Python rather than driven by
`data/configs/{dataset}_dataset.yaml` / `{dataset}_variants.yaml`.

**Headline finding: the claim holds for the core construction pipeline.**
`generic_data_manager.py`, `edge_factory.py`, `label_factory.py`,
`tag_constructor.py`, and `variant_registry.py` — the files that actually run
during training — contain **no hardcoded dataset-name branches**. Every
dataset-specific behavior (co-participation edge shape, N8 availability,
structural edges, M1 category scheme, M2 scalar source, z-score transform,
E11b's k) is dispatched through a config key: `has_secondary_id`,
`secondary_id_is_list`, `has_structural_edges`, `m1_meta_url`, `m2_source`,
`scalar_label_transform`, `m2_transform`, `minimum_hub_size`, `k_structural`,
`embedding_prefix`, etc. Adding a 6th dataset that fits the existing schema
requires only a new YAML pair, not a Python edit — confirmed by Electronics
and Toys already reusing Amazon's dispatch branches (`has_secondary_id: true`)
with zero code changes.

The violations below are concentrated in **analysis / figure-generation
scripts** that sit downstream of the pipeline, not the pipeline itself.

---

## CRITICAL — breaks plug-and-play claim

### 1. `code/decision_tree_analysis.py:434-436` and `:446`
```python
parser.add_argument('--dataset', choices=['history', 'amazon', 'arxiv', 'combined', 'all'], ...)
...
for ds in ['history', 'amazon', 'arxiv']:
```
Running `python3 code/decision_tree_analysis.py --dataset electronics` fails
immediately with an argparse error (`electronics` is not a valid choice).
Even `--dataset all` silently skips Electronics and Toys — `main()`'s data-
loading loop only iterates the 3-dataset literal above, never the 5 that
`experiment_runner.py` actually produces. The module's *programmatic* API
(`fit_tree`, `compare_trees`, `predict_and_validate`, used correctly by
`run_analysis.py`) is fully dataset-agnostic; only this file's own `main()`/
CLI is affected. **Fix:** derive the dataset list from
`Path('data/configs').glob('*_dataset.yaml')` instead of a literal.

---

## MODERATE — workaround exists, but real drift risk

### 2. Arxiv's M2/M6 exclusion lives in Python, duplicated in two files
- `code/run_analysis.py:61-63`: `PER_DATASET_EXCLUDE_TASKS = {'arxiv': {'M2', 'M6'}}`
- `code/generate_report_figures.py:52`: `PER_DATASET_EXCLUDE = {'arxiv': {'M2', 'M6'}}`

Same fact (arxiv's M2/centrality is ~91% predictable from degree — a label
leak), same value, two independent Python dicts under different names.
Nothing in `data/configs/arxiv_dataset.yaml` records this exclusion — a
reviewer reading only the YAML would never learn it exists, and the two
copies could silently drift apart on a future edit to only one file.
**Fix:** move to a YAML key, e.g. `exclude_tasks_from_analysis: [M2, M6]` in
`arxiv_dataset.yaml`, read once by both scripts.

### 3. `code/generate_report_figures.py:39` — `RUN_FINAL = Path('output/run_final')`
This directory does not exist anywhere in the repo (only `run_1000_final`,
`run_500_final`, `run_post_audit`, and a handful of dated `run_2026*` dirs
do). Every figure function in this file (`fig_01`...`fig_08`) will raise
`FileNotFoundError` on the first `load()` call. Not a dataset-name hardcode,
but relevant to Task 2's file-existence checks: none of this script's output
figures exist in the repo, and running it will not currently produce them.

### 4. `code/decision_tree_analysis.py:36-61` — `LABEL_MAP` / `FULL_LABEL`
```python
LABEL_MAP = {..., 'amazon': 'Amazon', 'arxiv': 'ArXiv', 'history': 'History'}
```
No `'electronics'` / `'toys'` entries. Any caller using
`LABEL_MAP.get(ds, ds)` directly (rather than `run_analysis.py`'s own
`_display()` helper, which falls back to `ds.title()`) prints the raw
lowercase dataset string for Electronics/Toys instead of a formatted name.
Cosmetic in the one place it's actually used today (workaround exists via
`_display()`), but the dict itself is stale relative to the 5-dataset scope.

---

## MINOR — cosmetic / doc-only

### 5. `code/edge_factory.py:187, 251, 310`
`_build_arxiv_coauthorship`, `_build_amazon_scalar_gt`,
`_build_history_neighbour_cooccurrence` are dispatched purely by config flag
(`secondary_id_is_list`, `has_secondary_id`, `has_structural_edges` — see
`_build_scalar_gt` at line 169) and contain no dataset-name checks in their
bodies. But Electronics and Toys already run through
`_build_amazon_scalar_gt` (their YAML sets `has_secondary_id: true`, same as
Amazon) under a function name that says "amazon." Purely a naming smell — a
future contributor could mistake this for an Amazon-only code path.

### 6. `code/variant_registry.py:21-22`
```python
Args:
    dataset: 'arxiv', 'amazon', or 'history'
```
Stale docstring — the constructor itself (`f"{dataset}_variants.yaml"`) has
no such restriction and works for all 5 datasets today.

### 7. `code/run_analysis.py:41-47` — `DATASET_RUNS` dict
Hardcodes a dataset→path mapping (all 5 currently point at
`output/run_1000_final`). Flagged for completeness only — the file's own
comment explicitly documents this as the intended single point of config
("change here, zero code changes elsewhere"), and it's overridable via
`--run-dir` at the CLI. Not an accidental hardcode.

---

## Not flagged (checked, found compliant)

- `generic_data_manager.py` — every dataset branch keys off `self.config[...]`
  values, never a dataset-name string. `node_type` branches (`N7`/`N8`/`N9`)
  are ontology dispatch, not dataset dispatch — out of scope for this audit.
- `edge_factory.py`'s `build_edges()` top-level dispatch — keys off
  `edge_type` (E10a...E11c), never dataset name.
- `label_factory.py`'s `generate_labels()` — M1 meta-join path is gated by
  `data_manager.config.get('m1_meta_url')`, not a dataset name; works
  identically for any dataset that sets the key (already shared by Amazon,
  Electronics, Toys).
- `tag_constructor.py` — no dataset references at all; `k_structural` is read
  from config with a Python-side default (`5`), which is a reasonable
  fallback pattern, not a hardcode of dataset-specific behavior.
- `experiment_runner.py`'s module-level `DATASETS = [...]` — a default list,
  fully overridable via `--datasets`, not a dispatch hardcode.
