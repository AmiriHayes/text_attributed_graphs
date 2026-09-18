#!/usr/bin/env python3
"""
format_rows.py — Stage 3 QA generation, Step 1: row formatter.

ArXiv only (first pass). Rows are already in the generic post-processing
schema (primary_id, secondary_id, aggregate_id, text_fidelity_a,
text_fidelity_b, categorical_label, scalar_label, structural_edges) —
NOT the raw Kaggle columns. text_fidelity_a = title+abstract (contextual),
text_fidelity_b = title only (compressed).
"""

MIN_CHARS = 100


def format_arxiv_row(row: dict) -> str | None:
    """
    Combine into a clean text string using text_fidelity_a (title+abstract).
    Returns None if the row fails the pre-filter (missing/blank text, or
    combined text under MIN_CHARS characters).
    """
    text = row.get('text_fidelity_a')
    if not isinstance(text, str):
        return None
    cleaned = ' '.join(text.split())  # collapse embedded newlines/whitespace
    if len(cleaned) < MIN_CHARS:
        return None
    return cleaned


if __name__ == '__main__':
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / 'data/arxiv/train/samples/sample_00.jsonl'
    rows = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            rows.append(json.loads(line))

    n_filtered = 0
    for i, row in enumerate(rows):
        out = format_arxiv_row(row)
        print(f'--- row {i} (primary_id={row.get("primary_id")}) ---')
        if out is None:
            print('  FILTERED (null/short text_fidelity_a)')
            n_filtered += 1
        else:
            print(f'  len={len(out)}  preview: {out[:200]}...')
        print()

    print(f'Filtered {n_filtered}/10')
