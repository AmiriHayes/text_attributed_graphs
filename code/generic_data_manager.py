"""
The single BaseDataManager implementation used for every dataset (arxiv,
amazon, history, electronics, toys). All dataset-specific behavior is driven
by {dataset}_dataset.yaml config keys (has_secondary_id, secondary_id_is_list,
has_structural_edges, m1_meta_url, m2_source, scalar_label_transform,
m2_transform, minimum_hub_size, embedding_prefix, etc.) rather than branching
on the dataset name — see docs/AUDIT_yaml_hardcoding.md for the full audit.

Reads:
  - data/configs/{dataset}_dataset.yaml
  - data/{dataset}/{split}/raw.jsonl, or
    data/{dataset}/{split}/samples/sample_{idx:02d}.jsonl
  - data/{dataset}/{split}/embeddings/{prefix}_embeddings[_contextual].npy
  - data/{dataset}/{split}/embeddings/{prefix}_index.json (N8/N9 only)

Writes:
  - Nothing.

Usage:
  Not run directly. Instantiated as GenericDataManager(dataset,
  base_path='data') by experiment_runner.py and every one-off script that
  needs dataset access.
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict, Counter

from base_data_manager import BaseDataManager


class GenericDataManager(BaseDataManager):
    """
    Unified data manager that works across arxiv, amazon, and history datasets.
    All dataset-specific behavior is driven by {dataset}_dataset.yaml config.
    """
    
    def __init__(self, dataset: str, base_path: str = "data"):
        """
        Initialize data manager.
        
        Args:
            dataset: 'arxiv', 'amazon', or 'history'
            base_path: Base directory containing data/ folder
        """
        self.dataset = dataset
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / dataset
        
        # Load dataset configuration
        config_path = self.base_path / "configs" / f"{dataset}_dataset.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Cache for loaded DataFrames and full embedding arrays
        self._cache = {}
        self._emb_cache = {}  # keyed by (node_type, fidelity, split) → full ndarray
    
    def load_data(self, split: str, sample_idx: Optional[int] = None) -> pd.DataFrame:
        """Load raw data for a given split."""
        cache_key = f"{split}_{sample_idx}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if sample_idx is not None:
            path = self.dataset_path / split / "samples" / f"sample_{sample_idx:02d}.jsonl"
        else:
            path = self.dataset_path / split / "raw.jsonl"
        
        df = pd.read_json(path, lines=True)
        self._cache[cache_key] = df
        return df
    
    def get_node_list(self, node_type: str, df: pd.DataFrame) -> List:
        """Get list of node identifiers for a given node type."""
        if node_type == 'N7':
            # Unique primary entities in df order (dict.fromkeys preserves insertion order).
            # Amazon has multiple reviews per product ASIN; returning duplicates would
            # cause shape mismatches downstream since graph nodes are deduplicated.
            return list(dict.fromkeys(df['primary_id'].tolist()))
        
        elif node_type == 'N8':
            # Secondary entities (authors/users)
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")

            # Count N7 occurrences per secondary_id so minimum_hub_size can
            # filter out non-aggregating "hub" nodes (pre-publication audit
            # FIX 4: 87-98% of N8 nodes have exactly 1 associated N7 node at
            # default settings -- functionally identical to N7, not a hub).
            counts: dict = {}
            for sec_id in df['secondary_id']:
                if sec_id is None:
                    continue
                # Handle list (arxiv) vs single value (amazon)
                ids = sec_id if isinstance(sec_id, list) else [sec_id]
                for sid in ids:
                    counts[sid] = counts.get(sid, 0) + 1

            min_hub_size = self.config.get('minimum_hub_size', 1)
            unique_ids = [nid for nid, c in counts.items() if c >= min_hub_size]

            return sorted(unique_ids)
        
        elif node_type == 'N9':
            # Aggregate entities (categories/families)
            return sorted(df['aggregate_id'].dropna().unique().tolist())
        
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_node_text(self, primary_id, fidelity: str, df: pd.DataFrame) -> str:
        """Get text for a node at specified fidelity level."""
        row = df[df['primary_id'] == primary_id]
        if len(row) == 0:
            return ""
        
        row = row.iloc[0]
        
        if fidelity == 'T12a':
            return row['text_fidelity_a'] or ""
        elif fidelity == 'T12b':
            return row['text_fidelity_b'] or ""
        elif fidelity == 'T12e':
            return ""  # T12e is always empty by definition
        else:
            raise ValueError(f"Unknown fidelity: {fidelity}")
    
    def get_embeddings(self, node_type: str, fidelity: str, split: str,
                      node_list: Optional[List] = None) -> np.ndarray:
        """Get embeddings for nodes at specified fidelity level."""
        emb_path = self.dataset_path / split / "embeddings"
        
        if node_type == 'N7':
            # Primary entity embeddings
            prefix = self.config['embedding_prefix']
            
            if fidelity == 'T12a':
                # Contextual embeddings
                if self.config['contextual_embedding_available']:
                    emb_file = emb_path / f"{prefix}_embeddings_contextual.npy"
                else:
                    # Fall back to T12b if contextual not available
                    emb_file = emb_path / f"{prefix}_embeddings.npy"
            elif fidelity == 'T12b':
                # Standard embeddings
                emb_file = emb_path / f"{prefix}_embeddings.npy"
            elif fidelity == 'T12e':
                prefix = self.config['embedding_prefix']
                ref = np.load(emb_path / f"{prefix}_embeddings.npy", mmap_mode='r')
                n_rows = len(node_list) if node_list is not None else len(self.load_data(split))
                return np.zeros((n_rows, ref.shape[1]), dtype=np.float32)
            else:
                raise ValueError(f"Unknown fidelity: {fidelity}")
            
            cache_key = (node_type, fidelity, split)
            if cache_key not in self._emb_cache:
                self._emb_cache[cache_key] = np.load(emb_file)
            embeddings = self._emb_cache[cache_key]

            if node_list is not None:
                full_df = self.load_data(split)
                pid_to_first_idx: dict = {}
                for i, pid in enumerate(full_df['primary_id']):
                    if pid not in pid_to_first_idx:
                        pid_to_first_idx[pid] = i
                indices = [pid_to_first_idx[pid] for pid in node_list
                           if pid in pid_to_first_idx]
                embeddings = embeddings[indices]
            
            return embeddings
        
        elif node_type == 'N8':
            # Secondary entity embeddings
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")

            prefix = self.config['secondary_embedding_prefix']
            emb_file = emb_path / f"{prefix}_embeddings.npy"
            index_file = emb_path / f"{prefix}_index.json"

            # T12e fix (pre-publication audit FIX 2): this branch previously
            # ignored `fidelity` entirely and always returned the real
            # embeddings, silently making T12e identical to T12b for every
            # N8 variant in every dataset. Mirror the N7 branch's explicit
            # zero-vector handling.
            if fidelity == 'T12e':
                ref = np.load(emb_file, mmap_mode='r')
                n_rows = len(node_list) if node_list is not None else ref.shape[0]
                return np.zeros((n_rows, ref.shape[1]), dtype=np.float32)

            cache_key = (node_type, fidelity, split)
            if cache_key not in self._emb_cache:
                self._emb_cache[cache_key] = np.load(emb_file)
            embeddings = self._emb_cache[cache_key]

            with open(index_file, 'r') as f:
                index = json.load(f)

            if node_list is not None:
                index_map = {nid: i for i, nid in enumerate(index)}
                indices = [index_map[nid] for nid in node_list if nid in index_map]
                embeddings = embeddings[indices]

            return embeddings

        elif node_type == 'N9':
            # Aggregate entity embeddings
            prefix = self.config['aggregate_embedding_prefix']
            emb_file = emb_path / f"{prefix}_embeddings.npy"
            index_file = emb_path / f"{prefix}_index.json"

            # T12e fix (pre-publication audit FIX 2): same bug as N8 above --
            # `fidelity` was never checked, so T12e silently equaled T12b.
            if fidelity == 'T12e':
                ref = np.load(emb_file, mmap_mode='r')
                n_rows = len(node_list) if node_list is not None else ref.shape[0]
                return np.zeros((n_rows, ref.shape[1]), dtype=np.float32)

            cache_key = (node_type, fidelity, split)
            if cache_key not in self._emb_cache:
                self._emb_cache[cache_key] = np.load(emb_file)
            embeddings = self._emb_cache[cache_key]

            with open(index_file, 'r') as f:
                index = json.load(f)

            if node_list is not None:
                index_map = {nid: i for i, nid in enumerate(index)}
                indices = [index_map[nid] for nid in node_list if nid in index_map]
                embeddings = embeddings[indices]

            return embeddings

        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_category_labels(self, node_type: str, df: pd.DataFrame,
                           node_list: Optional[List] = None) -> np.ndarray:
        """Get categorical labels for M1/M3 tasks."""
        if node_type == 'N7':
            # Build pid → first label mapping so Amazon (multiple reviews per
            # product) returns exactly one label per unique primary_id in node_list.
            pid_to_label: dict = {}
            for pid, lbl in zip(df['primary_id'], df['categorical_label']):
                if pid not in pid_to_label:
                    pid_to_label[pid] = lbl

            if node_list is None:
                node_list = self.get_node_list('N7', df)

            labels = np.array([pid_to_label.get(pid, np.nan) for pid in node_list])

            unique_labels = sorted(df['categorical_label'].dropna().unique())
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            n_classes = len(unique_labels)

            class_indices = np.array([label_to_idx.get(lbl, -1) for lbl in labels])
            one_hot = np.zeros((len(node_list), n_classes), dtype=np.float32)
            valid_mask = class_indices >= 0
            one_hot[valid_mask, class_indices[valid_mask]] = 1.0
            return one_hot
        
        elif node_type == 'N8':
            # Secondary entity labels: aggregate from primary entities
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")
            
            # Build mapping: secondary_id -> list of categorical_labels
            sec_to_labels = defaultdict(list)
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                cat_label = row['categorical_label']
                
                if sec_id is None or pd.isna(cat_label):
                    continue
                
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_labels[sid].append(cat_label)
                else:
                    sec_to_labels[sec_id].append(cat_label)
            
            # Compute mode (most common) label for each secondary entity
            if node_list is None:
                node_list = self.get_node_list('N8', df)
            
            mode_labels = []
            for nid in node_list:
                labels_list = sec_to_labels.get(nid, [])
                if labels_list:
                    mode_label = Counter(labels_list).most_common(1)[0][0]
                    mode_labels.append(mode_label)
                else:
                    mode_labels.append(None)
            
            # Convert to one-hot
            unique_labels = sorted(df['categorical_label'].dropna().unique())
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            n_classes = len(unique_labels)
            
            one_hot = np.zeros((len(node_list), n_classes), dtype=np.float32)
            for i, label in enumerate(mode_labels):
                if label is not None and label in label_to_idx:
                    one_hot[i, label_to_idx[label]] = 1.0
            
            return one_hot
        
        elif node_type == 'N9':
            # N9 categorical labels are tautological (node IS the category)
            # This should not be called for M1+N9 (excluded by remove list)
            raise ValueError("M1+N9 is tautological and should be excluded")
        
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_scalar_labels(self, node_type: str, df: pd.DataFrame,
                         graph=None, node_list: Optional[List] = None) -> np.ndarray:
        """Get scalar labels for M2/M4 tasks."""
        if node_type == 'N7':
            # Primary entity scalar labels
            if self.config.get('m2_source') == 'derived_from_text':
                # Abstract length: len(text_fidelity_a) - len(text_fidelity_b).
                # External to graph structure — no graph access needed.
                if node_list is None:
                    node_list = self.get_node_list('N7', df)
                pid_to_len: dict = {}
                for pid, tfa, tfb in zip(df['primary_id'],
                                         df['text_fidelity_a'],
                                         df['text_fidelity_b']):
                    if pid not in pid_to_len:
                        tfa = tfa if isinstance(tfa, str) else ''
                        tfb = tfb if isinstance(tfb, str) else ''
                        pid_to_len[pid] = len(tfa) - len(tfb)
                labels = np.array([float(pid_to_len.get(pid, 0.0)) for pid in node_list],
                                  dtype=np.float32)
                return labels.reshape(-1, 1)
            else:
                if node_list is None:
                    node_list = self.get_node_list('N7', df)
                if self.config.get('scalar_label_transform') == 'log1p':
                    # Mean log1p(scalar_label) across all rows sharing this primary_id.
                    # log1p before averaging: variance-stabilizes heavy-tailed counts
                    # (e.g. helpful_vote), consistent with N8/N9 treatment below.
                    pid_to_scalars: dict = defaultdict(list)
                    for pid, val in zip(df['primary_id'], df['scalar_label']):
                        if not pd.isna(val):
                            pid_to_scalars[pid].append(np.log1p(float(val)))
                    labels = np.array([
                        np.mean(pid_to_scalars[pid]) if pid_to_scalars.get(pid) else 0.0
                        for pid in node_list
                    ], dtype=np.float32)
                else:
                    # History: first-occurrence (primary_id unique per sample row).
                    pid_to_scalar: dict = {}
                    for pid, val in zip(df['primary_id'], df['scalar_label']):
                        if pid not in pid_to_scalar:
                            pid_to_scalar[pid] = val
                    labels = np.array([pid_to_scalar.get(pid, np.nan)
                                       for pid in node_list], dtype=np.float32)
                return labels.reshape(-1, 1)
        
        elif node_type == 'N8':
            # Secondary entity scalar labels: aggregate from primary entities
            if not self.config['has_secondary_id']:
                raise ValueError(f"N8 unavailable for {self.dataset}")
            
            if self.config.get('m2_source') == 'derived_from_text':
                # Mean abstract length across each author's papers in this sample.
                if node_list is None:
                    node_list = self.get_node_list('N8', df)
                author_to_lengths: dict = defaultdict(list)
                for sec_id, tfa, tfb in zip(df['secondary_id'],
                                            df['text_fidelity_a'],
                                            df['text_fidelity_b']):
                    if sec_id is None:
                        continue
                    tfa = tfa if isinstance(tfa, str) else ''
                    tfb = tfb if isinstance(tfb, str) else ''
                    abst_len = len(tfa) - len(tfb)
                    authors = sec_id if isinstance(sec_id, list) else [sec_id]
                    for author in authors:
                        author_to_lengths[author].append(abst_len)
                labels = np.array([np.mean(author_to_lengths.get(a, [0.0])) for a in node_list],
                                  dtype=np.float32)
                return labels.reshape(-1, 1)
            else:
                # Amazon: mean log1p(helpful_vote) of user's reviews
                sec_to_scalars = defaultdict(list)
                for _, row in df.iterrows():
                    sec_id = row['secondary_id']
                    scalar = row['scalar_label']

                    if sec_id is None or pd.isna(scalar):
                        continue

                    sec_to_scalars[sec_id].append(np.log1p(float(scalar)))
                
                if node_list is None:
                    node_list = self.get_node_list('N8', df)
                
                mean_scalars = np.array([
                    np.mean(sec_to_scalars.get(nid, [0.0])) for nid in node_list
                ], dtype=np.float32)
                
                return mean_scalars.reshape(-1, 1)
        
        elif node_type == 'N9':
            # Aggregate entity scalar labels
            if self.config.get('m2_source') == 'derived_from_text':
                # Mean abstract length across each category's papers in this sample.
                if node_list is None:
                    node_list = self.get_node_list('N9', df)
                cat_to_lengths: dict = defaultdict(list)
                for agg_id, tfa, tfb in zip(df['aggregate_id'],
                                            df['text_fidelity_a'],
                                            df['text_fidelity_b']):
                    if pd.isna(agg_id) or agg_id is None:
                        continue
                    tfa = tfa if isinstance(tfa, str) else ''
                    tfb = tfb if isinstance(tfb, str) else ''
                    cat_to_lengths[agg_id].append(len(tfa) - len(tfb))
                labels = np.array([np.mean(cat_to_lengths.get(c, [0.0])) for c in node_list],
                                  dtype=np.float32)
                return labels.reshape(-1, 1)
            else:
                # Amazon: mean log1p(helpful_vote); History: mean price (no transform).
                agg_to_scalars = defaultdict(list)
                for _, row in df.iterrows():
                    agg_id = row['aggregate_id']
                    scalar = row['scalar_label']

                    if pd.isna(agg_id) or pd.isna(scalar):
                        continue

                    val = np.log1p(float(scalar)) if self.config.get('scalar_label_transform') == 'log1p' else float(scalar)
                    agg_to_scalars[agg_id].append(val)
                
                if node_list is None:
                    node_list = self.get_node_list('N9', df)
                
                mean_scalars = np.array([
                    np.mean(agg_to_scalars.get(nid, [0.0])) for nid in node_list
                ], dtype=np.float32)
                
                return mean_scalars.reshape(-1, 1)
        
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def get_structural_edges(self, df: pd.DataFrame) -> List[Tuple]:
        """Get pre-given structural edges (E10c, History only)."""
        if not self.config['has_structural_edges']:
            raise ValueError(f"E10c structural edges unavailable for {self.dataset}")
        
        edges = []
        for _, row in df.iterrows():
            src = row['primary_id']
            targets = row['structural_edges']
            
            if targets is None:
                continue
            
            for tgt in targets:
                edges.append((src, tgt))
        
        return edges
    
    def get_config(self) -> Dict:
        """Get dataset configuration."""
        return self.config

# Made with Bob
