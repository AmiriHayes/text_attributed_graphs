#!/usr/bin/env python3
"""Rebuild every generated artifact in the paper, in dependency order.

    python3 paper/build_all.py            # everything
    python3 paper/build_all.py fig4 t3    # just those

Figure 1 is a hand-drawn diagram and is not generated here.

Prerequisite: output/run_final/ must be present.  It is produced by
code/experiment_runner.py (GNN scoring) and data/graphrag/ (question
generation and RAGAS scoring); see README.md.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'output' / 'run_final'

# key -> (paper artifact, script, args)
STEPS = [
    ('t3',   'Table 3  consistency',        'make_table3.py',           []),
    ('t4',   'Table 4  timing',             'make_table4.py',           []),
    ('fig2', 'Figure 2  Amazon 2x2',        'make_fig2_amazon.py',      []),
    ('fig3', 'Figure 3  Amazon trees',      'make_trees.py',            []),
    ('fig4', 'Figure 4  stability',         'make_fig4_stability.py',
     ['--include_control',
      '--out', str(RUN / 'analysis' / 'dt_consistency_with_control')]),
    ('fig5', 'Figure 5  construction',      'make_fig5_construction.py', []),
]
# fig3 and fig6 come from one script
ALIASES = {'fig6': 'fig3'}


def run(key, label, script, args):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)] + args,
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f'[{"ok " if ok else "FAIL"}] {label:28s} {script:26s} {time.time()-t0:5.1f}s')
    if not ok:
        print(r.stdout + r.stderr)
    return ok


# Every generator writes into paper/artifacts/. Names here are exactly the
# \includegraphics targets in the .tex, so a missing entry is a broken figure.
FIGURES = ['fig_amazon_combined_performance', 'fig_stability_dual'] + [
    f'fig_{d}_{t}_performance_runfinal'
    for d in ['arxiv', 'electronics', 'toys', 'history']
    for t in ['construction', 'graphrag']]


def collect():
    """Confirm every figure the manuscript includes was actually written."""
    dst = HERE / 'artifacts'
    missing = [n for n in FIGURES if not (dst / f'{n}.pdf').exists()]
    print(f'{len(FIGURES) - len(missing)}/{len(FIGURES)} paper figures present '
          f'in {dst.relative_to(HERE.parent)}')
    for m in missing:
        print(f'  MISSING {m}.pdf')


if __name__ == '__main__':
    if not RUN.is_dir():
        sys.exit(f'missing {RUN} -- see README.md for how to produce it')
    want = {ALIASES.get(a, a) for a in sys.argv[1:]}
    steps = [s for s in STEPS if not want or s[0] in want]
    if want and not steps:
        sys.exit(f'unknown target(s): {", ".join(sorted(want))}')
    failed = [s[1] for s in steps if not run(*s)]
    collect()
    print(f'\n{len(steps) - len(failed)}/{len(steps)} artifacts rebuilt'
          + (f'; FAILED: {", ".join(failed)}' if failed else ''))
    sys.exit(1 if failed else 0)
