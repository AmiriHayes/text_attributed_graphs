# Reviewer pass — findings, numbers, and what to do with them

Working notes for the revision. Not part of the submission. Every number below
is reproducible from a script in `code/` and a CSV in
`output/run_final/analysis/`.

---

## Tier 0 — correctness, do these first

### 1. E4b and E4c build the same graph on 4 of 5 datasets

`code/detect_duplicate_edge_rules.py` hashes edge sets and finds `E10b+E10c`
colliding on **11 (dataset, node type) pairs** — every node type of Sports,
ArXiv, Electronics and Toys. Only History distinguishes them.

Cause: `edge_factory._build_scalar_gt` is documented as *"weighted
co-participation edges"*, not scalar similarity, so it resolves to the same
identifier as `E10c`. Table 1's "**E4b** Scalar edges | *e.g. similar citation
count*" does not describe what the code builds. Same failure mode `code/README.md`
already records for `E11c` ("byte-identical to E10a").

Consequences: the space is smaller than reported (26→20, 16→12); the two
one-hot columns are perfectly collinear so any split between them is noise
(Figure 3's Amazon `E4b==0 → 12.95` vs `==1 → 12.82`, and Toys GraphRAG
`61.34` vs `60.82`); every rank correlation gains a guaranteed-consistent pair.

**Fixing it improves your numbers.** `python3 code/dt_consistency.py --dedupe_edges`:

| task | dataset | n → | ρ → | TED z → |
|---|---|---|---|---|
| node clf | Sports | 26→20 | 0.939→0.933 | −4.08→**−0.19** |
| node clf | ArXiv | 16→12 | 0.958→**0.981** | −3.05→−2.86 |
| node clf | Electronics | 26→20 | 0.957→0.957 | −2.71→−2.96 |
| node clf | Toys | 26→20 | 0.873→**0.903** | −2.63→−2.41 |
| node clf | History | 8→8 | 0.945 | −1.97 |
| RAG | Sports | 26→20 | 0.814→0.809 | −1.58→**−2.96** |
| RAG | ArXiv | 16→12 | 0.840→**0.975** | −2.16→−2.86 |
| RAG | Electronics | 26→20 | 0.805→0.802 | −1.96→−1.75 |
| RAG | Toys | 26→20 | 0.941→0.919 | −3.71→−2.63 |
| RAG | History | 8→8 | 0.976 | −0.56 |

Mean ρ: node classification 0.934 → **0.944**; GraphRAG 0.875 → **0.896**.

One cost to decide deliberately: Sports node-classification TED goes 0.05 → 0.34
(z −4.08 → −0.19), i.e. its structural consistency was carried by the duplicate
and is now indistinguishable from noise. Sports GraphRAG moves the other way.

Actions: dedupe, regenerate Figures 3 and 6 and the variant counts in Tables 3
and 4, fix Table 1's description of E4b, and add a sentence in Appendix A.

### 2. Table 4's ratios cannot be derived from Table 4

ArXiv GNN: 4.15 hr ÷ 3.12 min = **80×**, printed as **2.7×**. ArXiv RAG:
2.82 hr ÷ 6.41 min = **26.4×**, which matches. The GNN column silently uses the
5-subset plateau; the RAG column uses the full 300-question budget.

The derived sentence is also wrong: "153/4.0× = 38 GNN variants … 102/22.8× =
4.5 GraphRAG variants" divides by a ratio whose denominator is the **sum of five
baseline runs**. Correct per-dataset figures, already in your
`timing_table_plateau.csv`: **4.2–13.1** GNN variants per full-scale run
(`gnn_speedup_at_plateau`), and **0.6–1.1** for GraphRAG
(`gra_scale_matched_ratio`) — about one variant per run, not 4.5.

Fix: paste back the caption `paper/artifacts/table4_timing.tex` already
generates. It contains both missing disclosures.

### 3. Table 3 and Appendix A report different numbers

Table 3 is the `--include_control` run, Appendix A is the control-excluded run.
Toys GNN 0.95 vs 0.873; ArXiv TED 0.09 vs 0.045; n 24 vs 16. `dt_consistency.py`
says in a comment: *"RAISES mean within-dataset rho from 0.93 to 0.97 — report
both if this is on."* Also, §4.3's "every diagonal correlation above 0.9" is
false for the control-excluded Toys value (0.873).

Fix: state which run each table uses; give both means.

### 4. Two stripped qualifiers

- Table 3's generated caption says *"after Bonferroni correction over the 25
  cells of each panel"* — dropped from the manuscript. You did the correction.
- Table 4's generated caption (above) — dropped.

### 5. TED has a null you computed and didn't report

`analysis/dt_consistency/ted_random_baseline.csv` z-scores. As published,
GraphRAG History (−0.56) and Sports (−1.58) do not clear it. Add a z column.

### 6. Two dangling forward references

§3.1 and §3.2.2 both defer to Appendix A for content that isn't there, and the
Reproducibility Statement claims §3 gives the scoring formula in full — it
never appears. Draft supplied: `paper/drafts/appendix_a1_construction_details.tex`.

---

## Tier 1 — results you already have the data for

### 7. The tree matches the no-tree floor

`code/tree_ablation.py`:

| task | ρ tree | ρ no tree | ρ leave-one-variant-out |
|---|---|---|---|
| node classification | 0.934 | **0.939** | 0.802 |
| GraphRAG | 0.875 | **0.858** | 0.692 |

Table 3 measures the reproducibility of the proxy scores, not the value of the
meta-learner. Report the no-tree column and reframe the tree as the
interpretability layer; LOVO is the genuine generalization claim (ranking a
construction never scored). Two minutes of compute.

### 8. The practitioner payoff — zero new compute

| task | dataset | blind | default N1+T6a | TACO | oracle | gain recovered |
|---|---|---|---|---|---|---|
| node clf | History | 33.81 | 34.88 | 37.34 | 37.34 | 100% |
| node clf | Sports | 27.99 | 41.00 | 40.17 | 42.11 | 86% |
| node clf | ArXiv | 63.61 | 54.96 | **84.40** | 84.57 | 99% |
| node clf | Electronics | 36.29 | 54.33 | 56.12 | 56.74 | 97% |
| node clf | Toys | 7.76 | 11.25 | 12.14 | 12.53 | 92% |
| RAG | History | 58.20 | 57.91 | 66.07 | 68.08 | 80% |
| RAG | Sports | 68.30 | 71.16 | 71.36 | 73.19 | 63% |
| RAG | ArXiv | 66.58 | 66.63 | 68.43 | 69.46 | 64% |
| RAG | Electronics | 65.46 | 67.19 | 69.30 | 69.30 | 100% |
| RAG | Toys | 64.08 | 65.21 | 73.32 | 73.32 | 100% |

Matches or beats the untuned default on 9 of 10 pairs. Better abstract sentence
than ρ > 0.9. LaTeX: `python3 paper/make_review_tables.py`.

---

## Tier 2 — the subsampling confound, now measured

### 9. Subsets are not scale models of the full graph

`code/scale_diagnostics.py`, History:

| edge rule | edges/node @1k | @full pool | ratio |
|---|---|---|---|
| E4b | 9.14 | 162.7 | **17.8×** |
| E4c | 0.35 | 6.30 | **17.9×** |
| E5a | 0.03 | 0.34 | 10.5× |
| E5b | 4.37 | 4.43 | **1.01×** |

Attribute-match rules scale with pool size; k-NN does not. At subset scale
ArXiv N1 graphs are **90.8% isolated nodes**, N2 graphs **8.4%**.

### 10. E5b mostly isn't measuring structural similarity

E5b = k-nearest by degree-centrality difference, centrality computed over the
**E4b graph** (so it is not independent of the E4 axis). Fraction of nodes tied
at centrality 0, measured at 1k-row subset scale:

Sports N2 95.7% · ArXiv N1 91.2% · Toys N2 90.3% · Electronics N2 88.6% ·
Sports N1/N3 79.6/79.4% · Electronics N1/N3 74.8/74.1% · Toys N1/N3 62.7/62.4% ·
History N1 42.7% · **ArXiv N2 9.0%**

Among tied nodes the rule picks 5 arbitrary neighbours — hence the constant
~4.9 edges/node, 0% isolated, on every dataset at both scales. `edge_factory.py`
flags this and asks for replication on Amazon N8 ("ties most severe, 100%");
that replication never happened, and Sports N2 is where your main-body GraphRAG
tree roots on E5b.

### 11. Class priors do NOT explain the node-type effects

`analysis/full_scale/node_type_label_stats.csv` (full pool, via `LabelFactory`):

| dataset | node | nodes | classes | majority |
|---|---|---|---|---|
| History | N1 | 20,775 | 12 | 52.8% |
| Sports | N1/N2/N3 | 40,146/13,319/35,540 | 47 | 15.0/18.7/15.0% |
| ArXiv | N1/N2 | 50,000/89,626 | 18 | 20.9/24.3% |
| Electronics | N1/N2/N3 | 14,881/6,499/11,745 | 11 | 36.5/**51.5**/38.1% |
| Toys | N1/N2/N3 | 15,921/8,759/13,945 | 16 | 38.8/44.6/38.5% |

ArXiv's priors are near-identical yet N2 wins by 20+ points; Electronics N2 has
the better prior yet loses by ~30. So the node-type effect is not imbalance —
connectivity (item 9) is the live explanation. Table:
`paper/drafts/table_nodestats.tex`.

**ArXiv has 18 classes, not the 17 Appendix A states.** All 18 are populated
(smallest 238 nodes), there are no unmapped rows and no "other" bucket. They are
the standard arXiv primary archives: astro-ph, cond-mat, cs, gr-qc, hep-ex,
hep-lat, hep-ph, hep-th, math, math-ph, nlin, nucl-ex, nucl-th, physics, q-bio,
q-fin, quant-ph, stat. The appendix's description of ArXiv as "computer science,
mathematics, and physics preprints" also omits q-bio, q-fin and stat.

---

## Tier 3 — full-scale validation (`code/full_scale_score.py`)

The experiment the central claim needs, now run on four of five datasets
(Sports was not reached). Cost was not prohibitive: History 6.4 min, Toys
29 min, Electronics ~20 min, ArXiv 74 min.

### The core claim validates

| dataset | variants | over 200k guard | ρ subset→full | ρ tree→full | gain recovered at full scale | mean subset | mean full |
|---|---|---|---|---|---|---|---|
| History | 8 | 2 † | 0.829 | 0.488 | 100% | 33.2 | 55.5 |
| ArXiv | 16 | 10 † | 0.776 | 0.823 | 100% | 63.1 | 74.4 |
| Electronics | 20 | 4 | **0.902** | **0.915** | 95.5% | 38.1 | 63.2 |
| Toys | 20 | 6 | 0.856 | 0.854 | 81.4% | 6.0 | 48.1 |

† the proxy's top-ranked construction is itself over the guard.

Mean ρ(subset ranking → full-scale performance) = **0.84**. The tree's
top-ranked construction recovers **81–100%** of the oracle gain measured at
full scale. This is the result the paper asserts on every page and has never
shown. Put it in the abstract.

### Caveat 1 — direction survives, magnitude does not

Node-type ordering is preserved everywhere. The size of the effect is not.
Ratio of best to worst node type:

| dataset | subset scale | full scale |
|---|---|---|
| ArXiv | 1.73× | **1.08×** |
| Electronics | 4.50× | **2.02×** |
| Toys | 8.02× | **1.16×** |

So §4.2's "selecting reviewer nodes destroys signal beyond recovery" is a
statement about 1,000-row subsets. At full scale Toys N2 is 14% below N1, not
8× below. Mechanism is item 9.

Everything also rises sharply at full scale (Toys 6.0 → 48.1). The absolute
ranges in Appendix A ("Toys 1.7 to 12.5", "ArXiv 38.3 to 84.6") are properties
of the proxy, not the datasets. Part of that shift is estimator noise, not just
more training data: McFadden pseudo-R² is computed against an intercept-only
model fit on the test mask's own class frequencies, which with 150 test nodes
and 47 classes is very unstable and with 7,500 is not.

### Caveat 2 — the proxy ranks constructions that cannot be built

Variants over the pipeline's own `MAX_EDGES = 200k` guard at full scale:
History 2/8, **ArXiv 10/16**, Electronics 4/20, Toys 6/20. On History and ArXiv
the *top-ranked* construction is one of them — History's tree roots on
`E4b == 1 → 39.14`, which is 3,379,378 edges on 20,775 nodes at full scale.

Nothing currently warns a practitioner that a recommended construction is
undeployable. A feasibility column in the ranking output would fix it.

### Note on the duplicate-edge finding

E4b and E4c are byte-identical at **1,000-row subset scale**, which is where
every published number comes from. At full scale they stay identical on ArXiv
N1 (125,858 both) but diverge trivially on ArXiv N2 (774,187 vs 774,165, 0.003%).
Phrase the claim as "identical at the scale all variants are scored at."

## Smaller items

- Figure 2 plots `Top1` accuracy ×100; Figure 3's trees plot `S_GNN_step1`
  (floored McFadden) ×100. Same page, two metrics, neither labelled. (Distinct
  from the T6e issue in item 3.)
- `\url{ https://…}` — leading space swallowed; PDF reads "available
  athttps://…-C7ECas anonymous".
- Ω notation: reintroduce Ω vs Ω_valid; define the product at the end of §3 and
  the valid subset in the pruning-constraints paragraph.
- Figure 2 caption: say "three subsets and six variants shown for legibility".
- Rename §5 "Discussion, Limitations and Conclusion".
- Overclaiming: drop only the adverb in "unambiguously demonstrate", and
  "proving" → "indicating" for the task–dataset bound (rests on two datasets).
- Bib: `franceschi2020` is ICML 2019 (key/year disagree); `\citet`→`\citep`;
  `mcauley2023`/2024 and `guo2024lightrag`/2025 key-year mismatches;
  `hutter2019automl` cited twice for specific empirical claims — give a chapter
  or drop.
- Question gate is **per-subtype**, not per-question: History is 100/100/33, all
  others 100/100/100, so one subtype was dropped wholesale. The paper describes
  per-question filtering. One-sentence fix. The gate runs on a
  construction-agnostic flat vector index — say so, it's a point in your favour.
- `timing_table_raw.csv` has no generator in the repo, yet `base_gnn_s` is
  Table 4's most load-bearing column.
- Repo hygiene: six files end `# Made with Bob`; no LICENSE; `requirements.txt`
  unpinned, omits `ragas`, header still says "Ontological Generalization
  Framework"; `code/graphrag/` has 43 scripts with heavy near-duplication and
  the README names three.

---

## New scripts

| script | produces |
|---|---|
| `code/detect_duplicate_edge_rules.py` | `analysis/edge_rule_duplicates.csv` |
| `code/dt_consistency.py --dedupe_edges` | `analysis/dt_consistency_dedup/` (default path unchanged; `test_dt_consistency.py` still passes 0 failures) |
| `code/tree_ablation.py` | `analysis/tree_ablation/tree_ablation.csv` |
| `code/scale_diagnostics.py` | `analysis/full_scale/edge_density_by_scale.csv`, `node_type_label_stats.csv` |
| `code/full_scale_score.py` | `full_scale/full_scale_scores_{ds}.csv` |
| `code/full_scale_validation.py` | `analysis/full_scale/full_scale_validation.csv` |
| `code/architecture_sensitivity.py` | `analysis/architecture/arch_agreement.csv` |
| `paper/make_review_tables.py` | `table5_ablation.tex`, `table6_payoff.tex`, `table7_full_scale.tex` |

Drafts: `appendix_a1_construction_details.tex`, `limitations_paragraph.tex`,
`table_nodestats.tex`.
