# code/

The TACO-Graph pipeline. Nothing here produces a paper figure or table; that
is `paper/`.

## Pipeline

| Script | Role |
|---|---|
| `experiment_runner.py` | Entry point for GNN scoring. Trains every valid variant on every subset and writes `construction_performance_table_<dataset>.csv`. |
| `tag_constructor.py` | Assembles a PyG graph from a `(N, E, T)` variant. |
| `edge_factory.py` / `label_factory.py` | Build the edge set and the task label for a variant. |
| `models.py`, `trainer.py`, `global_trainer.py` | GNN definition and training loop. |
| `base_data_manager.py`, `generic_data_manager.py` | Load a dataset's rows, samples and embeddings. |
| `variant_registry.py` | Reads `data/configs/<dataset>_variants.yaml` and enumerates the valid variants. |
| `characterize_dataset.py`, `preprocess_dataset.py` | Dataset validity checks and the meta joins some datasets need. |
| `helpers/extend_samples.py` | Generates subsets 50-74, giving the 75 per split the paper uses. |
| `graphrag/` | Question generation and RAGAS scoring. See its README. |

## Analysis

| Script | Role |
|---|---|
| `dt_consistency.py` | **The published estimator.** Held-out rho and tree edit distance per task and dataset. Feeds Table 3 and Figure 4. |
| `build_table5_timing.py` | Plateau detection and timing, feeding Table 4. |
| `generate_ablation_heatmaps.py` | Heatmap helper used by the figure code. |
| `test_dt_consistency.py` | Verification suite: determinism, pool integrity, leakage, permutation nulls, TED sanity. Run it after any analysis change. |

`decision_tree_analysis.py` is **superseded by `dt_consistency.py`** and kept
only because the retired outputs in `output/run_final/analysis/superseded/`
came from it. It uses `min_samples_leaf=3`, which makes one-hot edge splits
unreachable once the no-text control is dropped, exposes only three of the
five datasets in its CLI, and reads a path that no longer exists. Do not use
it for new work.

## Naming

Internal variant codes are six higher than the paper's: `N7/E10b/T12a` here is
`N1/E4b/T6a` in the manuscript. `paper/_common.relabel` does the conversion on
the way out, so raw CSVs and the paper disagree by design.

Edge vocabulary, which is inconsistent in older comments:

| Internal | Paper | Meaning |
|---|---|---|
| `E10a` | `E4a` | categorical ground-truth edges |
| `E10b` | `E4b` | scalar ground-truth edges |
| `E10c` | `E4c` | participation edges |
| `E11a` | `E5a` | semantic similarity (cosine on text embeddings) |
| `E11b` | `E5b` | structural similarity (k-NN on centrality, k=5) |
| `E11c` | — | deprecated; was byte-identical to `E10a` (see `docs/AUDIT_pre_publication.md`) |

**`E10d` does not exist.** `edge_factory.py` has no `E10d` branch, so no
`E10d` variant can be built. Some older comments in
`data/configs/history_dataset.yaml`, `characterize_dataset.py` and
`code/graphrag/` use `E10d` loosely for History's participation edges; those
are `E10c`.
