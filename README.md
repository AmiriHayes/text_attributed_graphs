# TAG Ontological Generalization Framework Research

Research for systematic exploration of Text-Augmented Graph (TAG) variants using an **Ontological Generalization Framework**. This system implements Factory and Builder design patterns to scale from dozen to thousands of graph learning experiments @ NJIT Fall '25, Advisor: Kristina Wicke.

Latest Poster: [Link Here](https://drive.google.com/file/d/1qWZzVyVS7GusbsD7BZXw6rxxHbufvlQ7/view?usp=sharing)

Amiri's Fall Report: [Link Here](https://drive.google.com/file/d/1dAJoczUQUu_gy-kUDrsbtnQ4bQ4pC5yo/view?usp=sharing)

Research Notebook: [Link Here](https://drive.google.com/file/d/1qH5AGSc__xBFFu91OIY_U0_JcyhC_I68/view?usp=sharing)

## Repository layout

```
code/          the TACO-Graph pipeline: sampling, construction, scoring, analysis
  graphrag/    question generation and RAGAS scoring
paper/         generators for every figure and table in the manuscript
  artifacts/   their output
data/          inputs only: configs, raw rows, sampled subsets, questions
output/
  run_final/   the canonical run every paper number comes from
  figures/     the 10 figures the paper includes
```

## Reproducing the paper

```bash
python3 paper/build_all.py        # every generated figure and table, ~25s
python3 code/test_dt_consistency.py   # verification suite for the analysis
```

Both read `output/run_final/`, which is committed. To rebuild that run from
raw data (hours, and a GPU helps):

```bash
python3 code/experiment_runner.py --datasets amazon arxiv history electronics toys
python3 code/graphrag/extend_questions_to_300.py     # needs OPENAI_API_KEY
python3 code/graphrag/run_phase3b_graphrag_eval.py
python3 code/dt_consistency.py --include_control
```
