# paper/

Everything that turns the canonical run into a figure or table in the paper.
Nothing here trains, scores, or samples anything; that is `code/`.

```bash
python3 paper/build_all.py          # all 6 generated artifacts, ~25s
python3 paper/build_all.py fig4 t3  # just those
```

Outputs land in `paper/artifacts/`, named after the `\includegraphics`
targets in the manuscript.

| Paper artifact | Script | Output |
|---|---|---|
| Figure 1 | hand-drawn, not generated | `centerpiece_tag_paper.pdf` (external) |
| Figure 2 | `make_fig2_amazon.py` | `fig_amazon_combined_performance.pdf` |
| Figure 3 | `make_trees.py` | `fig3_trees_amazon.tex` |
| Figure 4 | `make_fig4_stability.py` | `fig_stability_dual.pdf` |
| Figure 5 | `make_fig5_construction.py` | 8 × `fig_<dataset>_<task>_performance_runfinal.pdf` |
| Figure 6 | `make_trees.py` | `fig6_trees_appendix.tex` |
| Table 3 | `make_table3.py` | `table3_consistency.tex` |
| Table 4 | `make_table4.py` | `table4_timing.tex` |

`_common.py`, `_construction_panel.py` and `_graphrag_panel.py` are helpers,
not entry points.

## Two conventions worth knowing

**Variant numbering.** Internally variants are `N7/E10b/T12a`; the paper uses
`N1/E4b/T6a`. `_common.relabel` subtracts six. Every artifact is relabelled on
the way out, so raw CSVs and the manuscript disagree by design.

**Rule A.** Figures 2 and 5 plot the best and worst variant per node type,
ranked by the GraphRAG composite, and reuse that one selection for both tasks
so a colour names the same construction in both panels. Defined once in
`_construction_panel.rule_a`; `_graphrag_panel` imports it rather than
re-deriving it. Ranking is on all 300 questions, not the test split; the two
bases disagree on three of five datasets.

## The no-text control

`T6e` (no text features) is a baseline, not a construction choice. It is
excluded from the trees in Figures 3 and 6, but **included** in Table 3,
Table 4 and Figure 4 so those three agree on variant counts. Including it
raises mean within-dataset rho from 0.93 to 0.97 and removes every negative
cross-dataset transfer on the GNN side, because all five datasets agree that
text beats no text. `make_table3.py --no-control` reproduces the conservative
version.

## Regenerating the run itself

`paper/` reads `output/run_final/` and never writes to it. To rebuild that
directory from raw data, see the root `README.md`.
