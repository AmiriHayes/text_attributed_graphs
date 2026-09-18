"""Shared helpers for the paper's artifact generators.

Every script in paper/ writes exactly one numbered artifact from the
canonical run in output/run_final/.  Nothing here trains or scores anything;
all of that lives in code/ and has already been run.
"""
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / 'code'
RUN = REPO / 'output' / 'run_final'
ART = Path(__file__).resolve().parent / 'artifacts'
sys.path.insert(0, str(CODE))

DATASETS = ['history', 'amazon', 'arxiv', 'electronics', 'toys']
LABEL = {'history': 'History', 'amazon': 'Amazon Sports', 'arxiv': 'ArXiv',
         'electronics': 'Electronics', 'toys': 'Toys'}
SHORT = {'history': 'Hist', 'amazon': 'Sports', 'arxiv': 'ArXiv',
         'electronics': 'Elec', 'toys': 'Toys'}
AXES = ['Node_Idx', 'Edge_Idx', 'Text_Idx']

# Paper numbering is the internal numbering minus six (N7->N1, E10a->E4a,
# T12a->T6a).  Anchored on separators because '\b' never fires inside
# 'N7_E10b_T12a' -- underscore is a word character.
_RELABEL = re.compile(r'(?:(?<=^)|(?<=[_/\s]))([NET])(\d+)([a-z]?)(?=[_/\s]|$)')


def relabel(text: str) -> str:
    return _RELABEL.sub(
        lambda m: f'{m.group(1)}{int(m.group(2)) - 6}{m.group(3)}', text)


def out(name: str) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    return ART / name
