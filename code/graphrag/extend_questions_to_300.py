#!/usr/bin/env python3
"""
extend_questions_to_300.py -- extends each dataset's questions.csv from the
current 83 single_specific / 33 aggregate_cross_paper / 34 edge_multihop
(150 total) to an even 100 / 100 / 100 (300 total), by APPENDING only.

Purpose: the 150-question sets are unevenly distributed across subtypes, so
a 75/75 split cannot be balanced per subtype. At 100/100/100 the 150/150
split carries 50 of each subtype per side.

Deficits are computed from the file's LIVE counts, not hardcoded, so a
partial or repeated run tops up rather than over-generating.

Generation order per spec: aggregate_cross_paper (most needed) ->
edge_multihop (most expensive) -> single_specific (cheapest).

PER-DATASET GENERATION CONFIG (GEN_CONFIG below)
Each dataset reuses the prompts/labels/validators from the script that
produced its ORIGINAL questions, not a one-size-fits-all fallback:
  arxiv                 run_question_generation.py            -- default
                        AGGREGATE_SYSTEM_PROMPT / 'Paper',
                        EDGE_MULTIHOP_SYSTEM_PROMPT, connector 'author'
  amazon                run_question_generation_amazon.py     -- AMAZON_*
                        prompts / 'Product', connector 'reviewer'
  electronics, toys     run_question_generation_amazon_schema.py (identical
                        schema to amazon: scalar secondary_id, aggregate_id
                        family grouping) -- same AMAZON_* config
  history               run_question_generation_history.py    -- HISTORY_*
                        prompts / 'Book'; edge_multihop via structural_edges
                        (E10d curated related-book links), NOT secondary_id,
                        because history_dataset.yaml has
                        has_secondary_id: false -- no author/reviewer hop
                        exists. Validated with validate_edge_multihop_titles.

  NOTE: extend_questions_to_150.py (Phase 0) imported the AMAZON_* prompts
  for every dataset, but it only ever exercised the single_specific path
  (all 48 of its additions were single_specific), so that generic fallback
  was never actually used for aggregate or edge_multihop. This script does
  exercise those paths, so it dispatches per dataset instead.

DEDUPLICATION (new here; Phase 0 did not need it)
Phase 0 added only single_specific, which dedupes naturally by excluding
already-used source_row_id. Generating 67 aggregate + 66 edge_multihop from
the same pool that already yielded 33 + 34 makes collisions likely, and
neither subtype has a single source_row_id to key on. So:
  single_specific  -- exclude pool rows already used as source_row_id
  edge_multihop    -- exclude already-used unordered hop_source_ids pairs
  aggregate        -- no source anchor exists; dedupe on normalized
                      question text
Normalized-question-text dedup is additionally applied to all three
subtypes, and within the new batch, as a backstop.

VALIDATION GATE
Same gate as Phase 0 (RAGAS faithfulness pilot > 0.5, NaN a hard failure,
same baseline VectorStoreIndex from evaluate_rag.build_index), but run
PER SUBTYPE rather than once on the first 8 of the whole batch. Under the
spec's generation order those first 8 would all be aggregate, gating 150
questions on one subtype's quality. Three gates per dataset; a subtype that
fails is dropped and reported, the others still land. The pool index is
built once per dataset and shared across the three gates.

Reads:
  - data/{dataset}/questions.csv (current 150)
  - data/{dataset}/train/samples/sample_00..09.jsonl (pool)

Writes:
  - data/{dataset}/questions.csv (300 rows; existing rows untouched)

Usage:
  python3 extend_questions_to_300.py --dataset arxiv
  python3 extend_questions_to_300.py --dataset history --skip_pilot_gate
"""
import argparse
import ast
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── ragas / langchain_community vertexai shim -- see
# extend_questions_to_150.py for the full explanation. ragas imports
# ChatVertexAI for a type reference it never instantiates here; current
# langchain-community no longer ships that module. Stubbing it out avoids a
# langchain-core downgrade cascade.
_dummy = types.ModuleType('langchain_community.chat_models.vertexai')
class _ChatVertexAIStub:
    pass
_dummy.ChatVertexAI = _ChatVertexAIStub
sys.modules.setdefault('langchain_community.chat_models.vertexai', _dummy)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_rows import format_arxiv_row
from generate_qa import (
    generate_qa, generate_aggregate_qa, generate_edge_multihop_qa,
    validate_qa, validate_edge_multihop, validate_edge_multihop_titles,
    classify_question_type, sample_cross_category_groups,
    sample_edge_multihop_pairs, sample_structural_edge_pairs, _client,
    AGGREGATE_SYSTEM_PROMPT, EDGE_MULTIHOP_SYSTEM_PROMPT,
    AMAZON_AGGREGATE_SYSTEM_PROMPT, AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
    HISTORY_AGGREGATE_SYSTEM_PROMPT, HISTORY_STRUCTURAL_EDGE_SYSTEM_PROMPT,
)
from evaluate_rag import load_pooled_samples, build_index, evaluate_with_ragas

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TARGET_PER_SUBTYPE = 100
SEED = 242                      # distinct from 42 (original) and 142 (Phase 0)
PILOT_N = 8
PILOT_FAITHFULNESS_THRESHOLD = 0.5
# edge_multihop gets the largest budget: the observed acceptance rate on
# history's structural_edges path is ~25% (validator + duplicate-pair
# rejections), so a 4x budget would leave no headroom for 66 accepts.
BUDGET_MULT = {'aggregate_cross_paper': 3, 'edge_multihop': 6, 'single_specific': 3}
SUBTYPE_ORDER = ['aggregate_cross_paper', 'edge_multihop', 'single_specific']
ID_PREFIX = {'single_specific': 'sh', 'aggregate_cross_paper': 'agg', 'edge_multihop': 'edge'}

_AMAZON_FAMILY = {
    'aggregate': dict(system_prompt=AMAZON_AGGREGATE_SYSTEM_PROMPT,
                      item_label='Product', text_fields=('text_fidelity_b',)),
    'edge': dict(mechanism='secondary_id', system_prompt=AMAZON_EDGE_MULTIHOP_SYSTEM_PROMPT,
                 anchor_label='Product 1 (anchor)',
                 target_label='Product 2 (connected via shared reviewer)',
                 connector_label='Shared reviewer identifier',
                 connector_words=['reviewer']),
}

GEN_CONFIG = {
    'arxiv': {
        'aggregate': dict(system_prompt=AGGREGATE_SYSTEM_PROMPT, item_label='Paper',
                          text_fields=('text_fidelity_b', 'text_fidelity_a')),
        'edge': dict(mechanism='secondary_id', system_prompt=EDGE_MULTIHOP_SYSTEM_PROMPT,
                     anchor_label='Paper 1 (anchor)',
                     target_label='Paper 2 (connected via shared author)',
                     connector_label='Shared author identifier',
                     connector_words=['author']),
    },
    'amazon': _AMAZON_FAMILY,
    'electronics': _AMAZON_FAMILY,
    'toys': _AMAZON_FAMILY,
    'history': {
        'aggregate': dict(system_prompt=HISTORY_AGGREGATE_SYSTEM_PROMPT, item_label='Book',
                          text_fields=('text_fidelity_b', 'text_fidelity_a')),
        'edge': dict(mechanism='structural_edges', system_prompt=HISTORY_STRUCTURAL_EDGE_SYSTEM_PROMPT,
                     anchor_label='Book 1',
                     target_label='Book 2 (documented related work)',
                     connector_label='Connection'),
    },
}


# ── dedup helpers ──────────────────────────────────────────────────────────
def _norm_q(q) -> str:
    """Normalized question text for duplicate detection: lowercase, strip
    punctuation, collapse whitespace."""
    if q is None or (isinstance(q, float) and pd.isna(q)):
        return ''
    s = re.sub(r'[^a-z0-9\s]', ' ', str(q).lower())
    return re.sub(r'\s+', ' ', s).strip()


def _parse_pair(val) -> frozenset | None:
    """hop_source_ids is stored as a string repr of a 2-element list. Returns
    an unordered pair, or None if unparseable."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, tuple)):
        items = list(val)
    else:
        try:
            items = ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return None
    if not isinstance(items, (list, tuple)) or len(items) != 2:
        return None
    return frozenset(str(x) for x in items)


# ── per-subtype generation ─────────────────────────────────────────────────
def gen_aggregate(ds, df, client, target, seen_q, seed=SEED):
    cfg = GEN_CONFIG[ds]['aggregate']
    budget = target * BUDGET_MULT['aggregate_cross_paper']
    groups = sample_cross_category_groups(df, n_groups=budget, group_size_range=(5, 10),
                                          min_categories=3, seed=seed)
    print(f'  sampled {len(groups)} candidate category-diverse groups (budget {budget})')
    results, rej, attempts = [], Counter(), 0
    for g in groups:
        if len(results) >= target or attempts >= budget:
            break
        attempts += 1
        cats = sorted(set(r['aggregate_id'] for r in g))
        try:
            qa = generate_aggregate_qa(g, client=client, **cfg)
        except ValueError as e:
            rej[f'API/parse: {type(e).__name__}'] += 1
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            rej[reason] += 1
            continue
        nq = _norm_q(qa['question'])
        if nq in seen_q:
            rej['duplicate_question_text'] += 1
            continue
        seen_q.add(nq)
        results.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'aggregate_cross_paper',
            'source_row_id': None, 'source_category_labels': cats,
            'source_text_preview': None, 'hop_type': 'single', 'hop_source_ids': None,
        })
    print(f'  aggregate_cross_paper: {len(results)}/{target} accepted in {attempts} attempts')
    return results, rej


def gen_edge(ds, df, client, target, seen_q, used_pairs, seed=SEED):
    cfg = dict(GEN_CONFIG[ds]['edge'])
    mechanism = cfg.pop('mechanism')
    connector_words = cfg.pop('connector_words', None)
    budget = target * BUDGET_MULT['edge_multihop']

    if mechanism == 'structural_edges':
        pairs = sample_structural_edge_pairs(df, n_pairs=budget, seed=seed)
        print(f'  sampled {len(pairs)} candidate structural_edges pairs (budget {budget})')
    else:
        pairs = sample_edge_multihop_pairs(df, n_pairs=budget, seed=seed)
        print(f'  sampled {len(pairs)} candidate shared-connector hop pairs (budget {budget})')

    results, rej, attempts = [], Counter(), 0
    for p in pairs:
        if len(results) >= target or attempts >= budget:
            break
        attempts += 1
        pair_key = frozenset((str(p['anchor']['primary_id']), str(p['target']['primary_id'])))
        if pair_key in used_pairs:
            rej['duplicate_hop_pair'] += 1
            continue
        at, tt = format_arxiv_row(p['anchor']), format_arxiv_row(p['target'])
        if at is None or tt is None:
            rej['null_text'] += 1
            continue

        shared = p.get('shared_author', 'documented related work')
        try:
            qa = generate_edge_multihop_qa(at, tt, shared, client=client, **cfg)
        except ValueError as e:
            rej[f'API/parse: {type(e).__name__}'] += 1
            continue

        if mechanism == 'structural_edges':
            valid, reason = validate_edge_multihop_titles(
                qa, p['anchor'].get('text_fidelity_b'), p['target'].get('text_fidelity_b'))
        else:
            valid, reason = validate_edge_multihop(qa, shared, connector_words=connector_words)
        if not valid:
            rej[reason] += 1
            continue

        nq = _norm_q(qa['question'])
        if nq in seen_q:
            rej['duplicate_question_text'] += 1
            continue
        seen_q.add(nq)
        used_pairs.add(pair_key)
        results.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'edge_multihop',
            'source_row_id': None, 'source_category_labels': None,
            'source_text_preview': None, 'hop_type': 'single',
            'hop_source_ids': [p['anchor']['primary_id'], p['target']['primary_id']],
        })
    print(f'  edge_multihop: {len(results)}/{target} accepted in {attempts} attempts')
    return results, rej


def gen_single(ds, df, client, target, seen_q, used_row_ids, seed=SEED):
    pool = df[~df['primary_id'].astype(str).isin(used_row_ids)]
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    budget = target * BUDGET_MULT['single_specific']
    print(f'  pool excl. {len(used_row_ids)} already-used rows: {len(pool)} available (budget {budget})')
    results, rej, attempts = [], Counter(), 0
    for _, row in shuffled.iterrows():
        if len(results) >= target or attempts >= budget:
            break
        attempts += 1
        text = format_arxiv_row(row.to_dict())
        if text is None:
            rej['formatter: null/short text'] += 1
            continue
        try:
            qa = generate_qa(text, client=client)
        except ValueError as e:
            rej[f'API/parse: {type(e).__name__}'] += 1
            continue
        valid, reason = validate_qa(qa)
        if not valid:
            rej[reason] += 1
            continue
        nq = _norm_q(qa['question'])
        if nq in seen_q:
            rej['duplicate_question_text'] += 1
            continue
        seen_q.add(nq)
        results.append({
            'question': qa['question'], 'reference_answer': qa['answer'],
            'question_type': classify_question_type(qa['question']),
            'question_subtype': 'single_specific',
            'source_row_id': row['primary_id'], 'source_category_labels': None,
            'source_text_preview': text[:100], 'hop_type': 'single', 'hop_source_ids': None,
        })
    print(f'  single_specific: {len(results)}/{target} accepted in {attempts} attempts')
    return results, rej


# ── pilot gate ─────────────────────────────────────────────────────────────
def pilot_gate(subtype, batch, index, pilot_n=PILOT_N) -> bool:
    pilot = batch[:pilot_n]
    print(f'  --- RAGAS pilot gate: {subtype} ({len(pilot)} of {len(batch)}) ---')
    pilot_q = [{'question': r['question'], 'reference_answer': r['reference_answer'],
                'question_type': r['question_type']} for r in pilot]
    scored = evaluate_with_ragas(pilot_q, index)
    mean_faith = scored['faithfulness'].mean()
    n_nan = int(scored['faithfulness'].isna().sum())
    print(f'    scores: {[round(x, 3) if pd.notna(x) else None for x in scored["faithfulness"].tolist()]}')
    print(f'    mean faithfulness: {mean_faith}  (threshold >{PILOT_FAITHFULNESS_THRESHOLD}, {n_nan}/{len(scored)} NaN)')
    # NaN is a hard failure, not a silent pass: nan <= threshold is False,
    # which would otherwise let a fully-broken scoring run through.
    if pd.isna(mean_faith) or n_nan > 0 or mean_faith <= PILOT_FAITHFULNESS_THRESHOLD:
        print(f'    GATE FAILED -- dropping all {len(batch)} {subtype} questions. '
              f'Reported rather than lowering the threshold or treating NaN as a pass.')
        return False
    print('    GATE PASSED.')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--skip_pilot_gate', action='store_true',
                    help='debug only -- accept batches without the RAGAS gate')
    ap.add_argument('--seed', type=int, default=SEED,
                    help='generation seed; change it for a genuinely fresh draw')
    ap.add_argument('--pilot_n', type=int, default=PILOT_N,
                    help='questions scored in the pilot gate. Raising this tightens the '
                         'estimate of the SAME batch mean against the SAME 0.5 threshold '
                         '(n=8 gives a standard error near 0.10, which is a coin flip for '
                         'a batch landing close to the line). It never weakens the rule.')
    args = ap.parse_args()
    ds = args.dataset
    if ds not in GEN_CONFIG:
        raise SystemExit(f'unknown dataset {ds!r}; expected one of {sorted(GEN_CONFIG)}')

    q_path = REPO_ROOT / f'data/{ds}/questions.csv'
    existing = pd.read_csv(q_path)
    n_original = len(existing)
    counts = existing['question_subtype'].value_counts().to_dict()
    print(f'=== {ds}: {n_original} existing questions ===')
    print(f'  current distribution: {counts}')

    deficits = {st: max(0, TARGET_PER_SUBTYPE - counts.get(st, 0)) for st in SUBTYPE_ORDER}
    print(f'  deficits to {TARGET_PER_SUBTYPE} each: {deficits}')
    if sum(deficits.values()) == 0:
        print('  nothing to do -- all subtypes already at target.')
        return

    # dedup state seeded from the existing file
    seen_q = {_norm_q(q) for q in existing['question']}
    seen_q.discard('')
    used_row_ids = set(existing['source_row_id'].dropna().astype(str))
    used_pairs = {pk for pk in (_parse_pair(v) for v in existing['hop_source_ids']) if pk}
    print(f'  dedup state: {len(seen_q)} question texts, {len(used_row_ids)} source rows, '
          f'{len(used_pairs)} hop pairs')

    max_idx = {}
    for st, pref in ID_PREFIX.items():
        mask = existing['id'].str.contains(f'_{pref}_', na=False)
        max_idx[st] = (existing.loc[mask, 'id'].str.extract(r'_(\d+)$')[0].astype(int).max()
                       if mask.any() else -1)

    client = _client()
    print(f'\nLoading {ds} pool (sample_00-09, deduped)...')
    df = load_pooled_samples(dataset=ds)
    print(f'  {len(df)} unique rows')

    batches, all_rej = {}, {}
    for st in SUBTYPE_ORDER:
        target = deficits[st]
        if target == 0:
            print(f'\n=== {st}: already at target, skipping ===')
            continue
        print(f'\n=== {st}: need {target} ===')
        if st == 'aggregate_cross_paper':
            res, rej = gen_aggregate(ds, df, client, target, seen_q, seed=args.seed)
        elif st == 'edge_multihop':
            res, rej = gen_edge(ds, df, client, target, seen_q, used_pairs, seed=args.seed)
        else:
            res, rej = gen_single(ds, df, client, target, seen_q, used_row_ids, seed=args.seed)
        batches[st], all_rej[st] = res, rej

    if not any(batches.values()):
        print('\nNo new questions generated -- aborting without writing.')
        return

    # Persist every generated batch BEFORE gating. A rejected batch previously
    # died with the process, which made a borderline gate result impossible to
    # re-examine or re-score without regenerating different questions
    # (generation runs at temperature 0.3, so no seed reproduces a batch).
    raw_dir = REPO_ROOT / 'data' / 'graphrag' / 'generated_batches'
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f'{ds}_seed{args.seed}_batches.json'
    with open(raw_path, 'w') as f:
        json.dump({st: b for st, b in batches.items()}, f, indent=2, default=str)
    print(f'\n  saved raw generated batches (pre-gate) -> {raw_path}')

    # ── pilot gates (index built once, shared across subtypes) ──
    accepted = {}
    if args.skip_pilot_gate:
        print('\n=== RAGAS PILOT GATES SKIPPED (--skip_pilot_gate) ===')
        accepted = {st: b for st, b in batches.items() if b}
    else:
        print('\n=== RAGAS PILOT GATES ===')
        print('  building baseline VectorStoreIndex over the pool (once, shared)...')
        index = build_index(df)
        for st, batch in batches.items():
            if not batch:
                continue
            if pilot_gate(st, batch, index, pilot_n=args.pilot_n):
                accepted[st] = batch

    if not accepted:
        print('\nAll subtype gates failed -- not writing to questions.csv.')
        return

    # ── assign continued ids, append ──
    new_rows = []
    for st in SUBTYPE_ORDER:
        if st not in accepted:
            continue
        pref = ID_PREFIX[st]
        idx = max_idx[st] + 1
        for r in accepted[st]:
            new_rows.append({'id': f'{ds}_{pref}_{idx:03d}', 'dataset': ds, **r, 'multi_hop_stub': None})
            idx += 1

    new_df = pd.DataFrame(new_rows, columns=existing.columns)
    combined = pd.concat([existing, new_df], ignore_index=True)
    assert combined['id'].is_unique, 'id collision -- refusing to write'
    combined.to_csv(q_path, index=False)

    print(f'\n=== REPORT: {ds} ===')
    for st in SUBTYPE_ORDER:
        gen_n = len(batches.get(st, []))
        acc_n = len(accepted.get(st, []))
        gate = 'n/a' if args.skip_pilot_gate else ('PASS' if st in accepted else 'FAIL')
        print(f'  {st:22s} target={deficits[st]:3d}  generated={gen_n:3d}  accepted={acc_n:3d}  gate={gate}')
        rej = all_rej.get(st)
        if rej:
            print(f'    rejections: {dict(rej)}')
    print(f'  original: {n_original}  added: {len(new_rows)}  final: {len(combined)}')
    print(f'  final distribution: {dict(combined["question_subtype"].value_counts())}')
    if len(combined) < 3 * TARGET_PER_SUBTYPE:
        print(f'  NOTE: {3 * TARGET_PER_SUBTYPE - len(combined)} short of {3 * TARGET_PER_SUBTYPE} -- '
              f'reported rather than padded with low-quality questions.')
    print(f'  saved -> {q_path}')


if __name__ == '__main__':
    main()
