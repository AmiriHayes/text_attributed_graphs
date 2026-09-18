"""
Minimal preprocess_dataset() -- the three things done by hand for the
Amazon-family datasets this project, generalized so any new Amazon-family
dataset gets them automatically: (1) secondary_id shape detection, (2) a
true-category meta join, (3) long-tail category consolidation.

Deliberately NOT the same algorithm as label_factory.py's production
_get_m1_category_map(): that function frequency-ranks AND collapses the
tail into "Other" in one pass (capped so Other stays <=5% of rows) --
verified against its cache (code/derived/{amazon,electronics,toys}_m1_*):
Amazon lands at 47 classes, Electronics at 11, Toys at 16, all well under
this module's default consolidation_threshold=50. Reusing that cache
directly would make Step 3 look like it never needs to run, when the real
question here is what the UNCAPPED category count looks like before any
consolidation choice is made. So Step 2 here does its own uncapped
frequency-rank join (every distinct l3_cat gets its own integer, no
Other bucket), cached separately under code/derived/preprocess_cache/,
and Step 3 does the top-N cut as its own explicit, configurable step.

Reads:
  - pool (directory of sample_*.jsonl, or a single CSV/JSONL -- same
    pooling convention as characterize_dataset.py, reused here)
  - the dataset's *_dataset.yaml config
  - the dataset's m1_meta_url (streamed, cached locally on first run)

Writes (dry_run=False only):
  - a NEW pool snapshot CSV (never the original raw files/CSVs -- that
    constraint is permanent, not just for this task)
  - YAML edits via text-surgery only (single-line value replacement for
    Step 1, block append for the `preprocessing:` section)

Usage:
    from preprocess_dataset import preprocess_dataset
    report = preprocess_dataset('electronics', 'data/electronics/train/samples',
                                 'data/configs/electronics_dataset.yaml',
                                 dry_run=True)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from characterize_dataset import _pool_samples  # reuse the same pooling convention

DEFAULT_CONSOLIDATION_THRESHOLD = 50
DEFAULT_CONSOLIDATION_TARGET = 20
PREPROCESS_CACHE_DIR = Path(__file__).parent / 'derived' / 'preprocess_cache'


# ── Step 1 — secondary_id type detection ────────────────────────────────

def _step1_secondary_id(df: pd.DataFrame, config: dict) -> dict:
    sample = df['secondary_id'].head(100) if 'secondary_id' in df.columns else pd.Series([], dtype=object)

    if len(sample) == 0 or sample.isna().all():
        inferred = {'has_secondary_id': False}
    else:
        non_null = sample.dropna()
        n_list = 0
        for v in non_null:
            if isinstance(v, list):
                n_list += 1
            elif isinstance(v, str) and any(c in v for c in ('[', '|', ',')):
                n_list += 1
        is_list = n_list > len(non_null) / 2  # majority vote
        inferred = {'has_secondary_id': True, 'secondary_id_is_list': is_list}

    current = {
        'has_secondary_id': config.get('has_secondary_id', False),
        'secondary_id_is_list': config.get('secondary_id_is_list', False),
    }

    mismatches = {}
    for k, v in inferred.items():
        if current.get(k) != v:
            mismatches[k] = {'yaml': current.get(k), 'inferred': v}

    return {
        'step': 1, 'name': 'Secondary ID type detection',
        'inferred': inferred, 'current_yaml': current,
        'status': 'MISMATCH' if mismatches else 'PASS',
        'mismatches': mismatches,
    }


# ── Step 2 — meta join for true categorical labels ──────────────────────

def _meta_cache_path(dataset: str) -> Path:
    PREPROCESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREPROCESS_CACHE_DIR / f'{dataset}_l3_uncapped.json'


def _fetch_uncapped_l3(dataset: str, meta_url: str, needed_asins: set) -> dict:
    """asin -> l3_cat, no Other-collapsing, no frequency cap. Cached to disk
    on first run (this is the expensive step -- a full stream of the meta
    file -- so it should never repeat for the same dataset)."""
    cache_path = _meta_cache_path(dataset)
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # cache is valid if it covers at least the asins we need now
        if needed_asins <= set(cached.keys()):
            return {k: cached[k] for k in needed_asins}

    import requests
    asin_to_l3: dict = {}
    found: set = set()
    with requests.get(meta_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pa = obj.get('parent_asin') or obj.get('asin')
            if pa in needed_asins and pa not in found:
                cats = obj.get('categories', [])
                l3 = (cats[2] if len(cats) > 2 else (cats[1] if len(cats) > 1 else None)
                      ) if isinstance(cats, list) else None
                asin_to_l3[pa] = l3
                found.add(pa)
            if len(found) >= len(needed_asins):
                break

    with open(cache_path, 'w') as f:
        json.dump(asin_to_l3, f)
    return asin_to_l3


def _step2_meta_join(df: pd.DataFrame, config: dict, dataset: str) -> dict:
    meta_url = config.get('meta_join_url') or config.get('m1_meta_url')
    prep = config.get('preprocessing', {})
    if not prep.get('preprocess_meta_join', False):
        return {'step': 2, 'name': 'Meta join for true categorical labels', 'status': 'SKIPPED',
                'reason': 'preprocess_meta_join not true in YAML'}
    if not meta_url:
        return {'step': 2, 'name': 'Meta join for true categorical labels', 'status': 'SKIPPED',
                'reason': 'no meta_join_url / m1_meta_url in YAML (required key missing)'}
    if 'aggregate_id' not in df.columns:
        return {'step': 2, 'name': 'Meta join for true categorical labels', 'status': 'SKIPPED',
                'reason': 'no aggregate_id column to join on (expected parent_asin/asin)'}

    needed = set(df['aggregate_id'].dropna().unique())
    asin_to_l3 = _fetch_uncapped_l3(dataset, meta_url, needed)

    l3_series = df['aggregate_id'].map(asin_to_l3)
    n_gained = int(l3_series.notna().sum())
    counts = Counter(l3_series.dropna())
    # frequency rank -> integer label, no cap, no Other bucket
    rank = {cat: i for i, (cat, _) in enumerate(counts.most_common())}
    new_labels = l3_series.map(rank)

    n_unique = len(rank)
    majority_frac = (counts.most_common(1)[0][1] / n_gained) if n_gained else float('nan')

    return {
        'step': 2, 'name': 'Meta join for true categorical labels', 'status': 'RAN',
        'rows_gained_category': n_gained, 'rows_total': len(df),
        'n_unique_categories': n_unique, 'majority_class_fraction': round(majority_frac, 4) if n_gained else None,
        '_new_categorical_label': new_labels,  # internal, used by dry_run=False write and by Step 3
    }


# ── Step 3 — category consolidation ──────────────────────────────────────

def _step3_consolidate(df: pd.DataFrame, config: dict, labels: Optional[pd.Series]) -> dict:
    prep = config.get('preprocessing', {})
    if not prep.get('preprocess_consolidate_categories', False):
        return {'step': 3, 'name': 'Category consolidation', 'status': 'SKIPPED',
                'reason': 'preprocess_consolidate_categories not true in YAML'}

    missing_keys = [k for k in ('consolidation_threshold', 'consolidation_target', 'consolidation_drop_tail')
                     if k not in prep]
    if missing_keys:
        return {'step': 3, 'name': 'Category consolidation', 'status': 'SKIPPED',
                'reason': f'required YAML key(s) missing: {missing_keys}'}

    threshold = prep['consolidation_threshold']
    target = prep['consolidation_target']
    drop_tail = prep['consolidation_drop_tail']

    active_labels = labels if labels is not None else (
        df['categorical_label'] if 'categorical_label' in df.columns else pd.Series([], dtype=object))
    active_labels = active_labels.dropna()
    n_unique = active_labels.nunique()

    if n_unique <= threshold:
        return {'step': 3, 'name': 'Category consolidation', 'status': 'SKIPPED',
                'reason': f'{n_unique} unique categorical_label values <= threshold {threshold}'}

    counts = active_labels.value_counts()
    keep = set(counts.head(target).index)
    n_total = len(active_labels)
    n_kept_rows = int(active_labels.isin(keep).sum())
    n_dropped_rows = n_total - n_kept_rows

    return {
        'step': 3, 'name': 'Category consolidation', 'status': 'RAN',
        'original_unique': int(n_unique), 'consolidated_unique': len(keep),
        'rows_retained_fraction': round(n_kept_rows / n_total, 4) if n_total else None,
        'drop_tail': drop_tail,
        'note': ('tail rows dropped entirely' if drop_tail else
                 'tail rows mapped to label -1 ("other"), not dropped'),
    }


# ── YAML helpers (text-surgery only) ─────────────────────────────────────

def _read_yaml_text(config_path: str) -> str:
    with open(config_path) as f:
        return f.read()


def _write_secondary_id_fix(config_path: str, mismatches: dict):
    text = _read_yaml_text(config_path)
    for key, vals in mismatches.items():
        new_val = 'true' if vals['inferred'] else 'false'
        pattern = re.compile(rf'^({re.escape(key)}:\s*)\S+', re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(lambda m: m.group(1) + new_val, text, count=1)
        else:
            if not text.endswith('\n'):
                text += '\n'
            text += f'{key}: {new_val}\n'
    with open(config_path, 'w') as f:
        f.write(text)


def ensure_preprocessing_block(config_path: str, defaults: dict):
    """Append the `preprocessing:` block via text-surgery if absent. Never
    touches existing content -- pure append, same pattern as
    characterize_dataset.py's dataset_characterization writer."""
    text = _read_yaml_text(config_path)
    if re.search(r'^preprocessing:', text, re.MULTILINE):
        return  # already present, leave as-is
    block = yaml.safe_dump({'preprocessing': defaults}, sort_keys=False, default_flow_style=False)
    if not text.endswith('\n'):
        text += '\n'
    text += block
    with open(config_path, 'w') as f:
        f.write(text)


# ── main entry point ─────────────────────────────────────────────────────

def preprocess_dataset(dataset_name: str, pool_path: str, config_path: str,
                        dry_run: bool = True, verbose: bool = True) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    df = _pool_samples(pool_path)

    s1 = _step1_secondary_id(df, config)
    s2 = _step2_meta_join(df, config, dataset_name)
    s3_labels = s2.get('_new_categorical_label')
    s3 = _step3_consolidate(df, config, s3_labels)

    files_written = []
    if not dry_run:
        if s1['status'] == 'MISMATCH':
            _write_secondary_id_fix(config_path, s1['mismatches'])
            files_written.append(config_path)
        if s2['status'] == 'RAN' or s3['status'] == 'RAN':
            out_df = df.copy()
            if s2['status'] == 'RAN':
                out_df['categorical_label'] = s2['_new_categorical_label'].values
            if s3['status'] == 'RAN':
                counts = out_df['categorical_label'].value_counts()
                keep = set(counts.head(config.get('preprocessing', {}).get(
                    'consolidation_target', DEFAULT_CONSOLIDATION_TARGET)).index)
                mask_tail = ~out_df['categorical_label'].isin(keep)
                if config.get('preprocessing', {}).get('consolidation_drop_tail', False):
                    out_df = out_df[~mask_tail]
                else:
                    out_df.loc[mask_tail, 'categorical_label'] = -1
            out_path = str(Path(pool_path).with_name(Path(pool_path).name + '_preprocessed.csv')) \
                if Path(pool_path).is_dir() else str(Path(pool_path).with_suffix('.preprocessed.csv'))
            out_df.to_csv(out_path, index=False)
            files_written.append(out_path)

    s2_report = {k: v for k, v in s2.items() if not k.startswith('_')}
    report = {
        'dataset': dataset_name,
        'step1': s1, 'step2': s2_report, 'step3': s3,
        'dry_run': dry_run, 'files_written': files_written,
    }
    if verbose:
        _print_report(report)
    return report


def _print_report(r: dict):
    print(f"\n{'='*70}\nDATASET: {r['dataset']}\n{'='*70}")
    s1, s2, s3 = r['step1'], r['step2'], r['step3']
    print(f"Step 1 secondary_id detection: {s1['status']}")
    print(f"    inferred={s1['inferred']}  current_yaml={s1['current_yaml']}")
    if s1['mismatches']:
        print(f"    mismatches: {s1['mismatches']}")

    if s2['status'] == 'RAN':
        print(f"Step 2 meta join: RAN ({s2['rows_gained_category']}/{s2['rows_total']} rows joined, "
              f"{s2['n_unique_categories']} categories, majority={s2['majority_class_fraction']})")
    else:
        print(f"Step 2 meta join: SKIPPED ({s2.get('reason')})")

    if s3['status'] == 'RAN':
        print(f"Step 3 consolidation: RAN ({s3['original_unique']}→{s3['consolidated_unique']} categories, "
              f"{s3['rows_retained_fraction']:.1%} rows retained, drop_tail={s3['drop_tail']})")
    else:
        print(f"Step 3 consolidation: SKIPPED ({s3.get('reason')})")

    print(f"DRY RUN: {'no files written' if r['dry_run'] else 'files written: ' + str(r['files_written'])}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('pool_path')
    ap.add_argument('config_path')
    ap.add_argument('--apply', action='store_true', help='dry_run=False')
    args = ap.parse_args()
    preprocess_dataset(args.dataset, args.pool_path, args.config_path,
                        dry_run=not args.apply)
