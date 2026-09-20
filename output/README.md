# output/

## `run_final/` — the canonical run

Every number in the paper comes from here and nowhere else. Treat it as
read-only input; `paper/build_all.py` reads it and never writes to it.

```
run_final/
  construction_performance_table_<dataset>.csv   GNN scores, one row per (variant, subset)
  ragas_results_<dataset>.csv                    RAGAS scores, one row per (variant, question)
  question_split_<dataset>.csv                   the frozen 150/150 question split
  epoch_logs/<dataset>/                          per-epoch curves from the main run
  epoch_logs_ruleA/<dataset>/                    per-epoch curves for the Rule A variants
  analysis/
    dt_consistency_with_control/   <- Table 3, Figure 4 (the published estimator)
    dt_consistency/                   the same analysis excluding the no-text control
    timing_table_{raw,plateau}.csv <- Table 4
    ted_random_baseline.csv           the null model quoted in Section 4.3
    superseded/                       retired analyses, kept for provenance only
```

**`analysis/superseded/` is not used by anything.** It holds earlier
cross-dataset matrices whose diagonals disagree with the published Table 3
because they used a different estimator (all tasks pooled with task as a
feature, `min_samples_leaf=3`, and no explicit no-text exclusion). They are
retained so the record is complete, not because any paper number depends on
them. If you want the published numbers, read
`analysis/dt_consistency_with_control/`.

## Figures and tables

Generated artifacts do not live here. `paper/build_all.py` writes them to
`paper/artifacts/`, named after the `\includegraphics` targets in the
manuscript.
