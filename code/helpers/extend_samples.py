#!/usr/bin/env python3
"""
Extend samples from 20 to 50 for all datasets.

Generates samples 20-49 using the same logic as the original generate_canonical_samples
function from write_data.ipynb. Uses random_state=i for sample i to ensure reproducibility.

Usage:
    python3 code/extend_samples.py
"""

import os
import json
from pathlib import Path
import pandas as pd

# Configuration
DATASETS = ['history', 'amazon', 'arxiv', 'electronics', 'toys']
SPLITS = ['train', 'test']
SAMPLE_SIZE = 1_000
START_IDX = 20  # Start from sample_20
END_IDX = 50    # Up to sample_49 (inclusive)

def generate_extended_samples(dataset: str, split: str):
    """
    Generate samples 20-49 for a given dataset and split.
    
    Reads from data/{dataset}/{split}/raw.jsonl and generates samples
    using the same seeding convention as the original samples (random_state=i).
    """
    # Paths
    raw_path = Path(f'data/{dataset}/{split}/raw.jsonl')
    samples_dir = Path(f'data/{dataset}/{split}/samples')
    
    if not raw_path.exists():
        print(f'  ⚠️  SKIP {dataset}/{split}: raw.jsonl not found')
        return 0
    
    # Load raw data
    print(f'  Loading {dataset}/{split}/raw.jsonl...')
    df = pd.read_json(raw_path, lines=True)
    print(f'    Loaded {len(df):,} rows')
    
    # Ensure samples directory exists
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples 20-49
    generated = 0
    for i in range(START_IDX, END_IDX):
        sample_path = samples_dir / f'sample_{i:02d}.jsonl'
        
        # Skip if already exists
        if sample_path.exists():
            print(f'    ⏭️  sample_{i:02d}.jsonl already exists, skipping')
            continue
        
        # Generate sample with random_state=i (same convention as original)
        sample = df.sample(
            n=min(SAMPLE_SIZE, len(df)),
            random_state=i
        ).reset_index(drop=True)
        
        # Save to JSONL
        sample.to_json(sample_path, orient='records', lines=True)
        generated += 1
    
    print(f'    ✅ Generated {generated} new samples (total now: {START_IDX + generated})')
    return generated


def main():
    print('='*72)
    print('EXTEND SAMPLES: Generate samples 20-49 for all datasets')
    print('='*72)
    
    total_generated = 0
    
    for dataset in DATASETS:
        print(f'\n{dataset.upper()}')
        print('-'*72)
        
        for split in SPLITS:
            generated = generate_extended_samples(dataset, split)
            total_generated += generated
    
    print('\n' + '='*72)
    print(f'DONE: Generated {total_generated} new sample files')
    print('='*72)
    
    # Verify all samples exist
    print('\nVERIFICATION:')
    print('-'*72)
    all_good = True
    for dataset in DATASETS:
        for split in SPLITS:
            samples_dir = Path(f'data/{dataset}/{split}/samples')
            if not samples_dir.exists():
                print(f'  ❌ {dataset}/{split}/samples directory not found')
                all_good = False
                continue
            
            missing = []
            for i in range(END_IDX):
                sample_path = samples_dir / f'sample_{i:02d}.jsonl'
                if not sample_path.exists():
                    missing.append(i)
            
            if missing:
                print(f'  ❌ {dataset}/{split}: Missing samples {missing}')
                all_good = False
            else:
                print(f'  ✅ {dataset}/{split}: All samples 00-49 present')
    
    if all_good:
        print('\n✅ All datasets have samples 00-49!')
    else:
        print('\n⚠️  Some samples are missing. Check output above.')


if __name__ == '__main__':
    main()

