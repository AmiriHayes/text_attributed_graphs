# code/graphrag/

Question generation and RAGAS scoring for the GraphRAG half of the pipeline.
Output lands in `output/run_final/` as `ragas_results_<dataset>.csv` and
`question_split_<dataset>.csv`.

## Live (8 files)

Run in this order; each needs `OPENAI_API_KEY`.

| Script | Role |
|---|---|
| `extend_questions_to_300.py` | Entry point. Grows each dataset's question set to 300, evenly split across the three subtypes, deduplicating on question text, hop pairs and source rows. Persists every batch before the faithfulness gate. |
| `build_canonical_split.py` | Entry point. Freezes the 150/150 train-test split and writes `question_split_<dataset>.csv`. Must run before `code/dt_consistency.py`. |
| `run_phase3b_graphrag_eval.py` | Entry point. Scores new questions only; resume is keyed on `(variant, question_id)` pairs, not row counts. |
| `generate_qa.py` | Prompts, validators and sampling helpers for all three question subtypes. |
| `evaluate_rag.py` | Builds the LlamaIndex retriever per variant and calls RAGAS. |
| `format_rows.py` | Turns a dataset row into the passage text a variant exposes. |
| `graphrag_split.py` | Single source of truth for the split (seed 42); frozen after first write. |
| `run_step2_4.py` | Shared driver used by the evaluation entry points. |

## Superseded (36 files)

Kept for provenance. Nothing in `code/`, `paper/` or the READMEs imports them,
and no published number depends on them.

They fall into five groups: the 150-question generation that preceded the
300-question set (`extend_questions_to_150.py`, the four
`run_question_generation*.py`, the five `gate_*.py`, `run_gate_v2.py`); the
per-dataset variant builders later unified (`build_all_variants*.py`,
`build_step1_graphs.py` and its `_v2`); the proxy-metric study that did not
make the paper (the six `proxy_*.py`); the N8 correction and stability
experiments (`rebuild_n8_corrected*.py`, `run_stability_study_step1.py`,
`analyze_stability_study.py`, `within_nodetype_analysis.py`); and one-off
checks (`arxiv_sanity_checks.py`, `build_e11b_k5_variant.py`,
`build_kg_baseline.py`, `run_full_ragas.py`,
`run_phase3_graphrag_eval.py`, superseded by `run_phase3b_`).
