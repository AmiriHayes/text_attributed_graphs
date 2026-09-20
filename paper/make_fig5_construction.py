#!/usr/bin/env python3
"""Figure 5 -- construction performance for the four appendix datasets, and
the two Amazon panels that Figure 2 is assembled from.

Each dataset produces two PDFs, one per task, with the SAME variants in the
same colours on both sides (Rule A: best and worst per node type ranked by
the GraphRAG composite, defined once in _construction_panel.rule_a).

Epoch-log source per dataset is resolved here rather than passed by hand:
Rule A selects variants that the main run did not log, so four datasets have
a side-run under output/run_final/epoch_logs_ruleA/.

Writes:  paper/artifacts/fig_{dataset}_{construction,graphrag}_performance_runfinal.pdf

Usage:   python3 paper/make_fig5_construction.py            # appendix four
         python3 paper/make_fig5_construction.py --all      # plus Amazon
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'output' / 'run_final'
APPENDIX = ['arxiv', 'electronics', 'toys', 'history']


def epoch_dir(ds: str) -> Path:
    """Rule A side-run if one exists, otherwise the main run's logs."""
    side = RUN / 'epoch_logs_ruleA' / ds
    return side if side.is_dir() else RUN / 'epoch_logs' / ds


def build(ds: str):
    for script, extra in [('_construction_panel.py', ['--epoch_dir', str(epoch_dir(ds))]),
                          ('_graphrag_panel.py', [])]:
        cmd = [sys.executable, str(HERE / script), '--dataset', ds] + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'FAILED {ds} {script}\n{r.stdout}{r.stderr}')
        print(f'  {ds:12s} ' + r.stdout.strip().splitlines()[-1].strip())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true',
                    help='also rebuild the Amazon panels used by Figure 2')
    a = ap.parse_args()
    for ds in (['amazon'] + APPENDIX if a.all else APPENDIX):
        build(ds)
