"""
Generates node/edge labels for a given task type (M1-M4; M5/M6 are derived
elsewhere, post-hoc, in trainer.py) by dispatching on task_type and node_type,
with M1's category scheme read from YAML (m1_meta_url) rather than hardcoded.

Reads:
  - code/derived/{dataset}_m1_category_map.json (cache, if present)
  - code/derived/{dataset}_m1_class_names.json (cache, if present)
  - data/{dataset}/{train,test}/raw.jsonl (streamed via data_root, only when
    the above cache is missing and data_manager.config has m1_meta_url set)
  - the dataset's m1_meta_url (streamed HTTP, only on a cache miss)

Writes:
  - code/derived/{dataset}_m1_category_map.json (cache, on a cache miss)
  - code/derived/{dataset}_m1_class_names.json (cache, on a cache miss)

Usage:
  Not run directly. Called by code/tag_constructor.py's
  TAGConstructor.construct() via LabelFactory.generate_labels(...).
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from collections import defaultdict, Counter

# ── Meta-join M1 derived-artifact directory (cache files are per-dataset) ─────
_DERIVED_DIR = Path(__file__).parent / "derived"


def _apply_local_zscore(labels: np.ndarray, node_list: list,
                        graph: nx.Graph) -> np.ndarray:
    """Transform raw scalar labels to local z-scores using graph neighborhoods.

    target_i = (label_i - mean_N(i)) / (std_N(i) + eps)

    eps = max(0.1 * global_label_std, 1e-8) prevents explosion when a node
    has only 1 neighbour (std=0) while preserving signal for dense constructions.
    Isolated nodes (no neighbours with valid labels) get target = 0.
    NaN labels are left as NaN — downstream masking handles them.
    """
    raw = labels.flatten().astype(np.float64)
    valid = np.isfinite(raw)
    global_std = float(np.std(raw[valid])) if valid.sum() > 1 else 1.0
    eps = max(0.1 * global_std, 1e-8)

    node_to_idx = {nid: i for i, nid in enumerate(node_list)}
    result = np.full(len(node_list), np.nan, dtype=np.float64)

    for i, nid in enumerate(node_list):
        if not valid[i]:
            continue
        nbr_idxs = [node_to_idx[v] for v in graph.neighbors(nid)
                    if v in node_to_idx and valid[node_to_idx[v]]]
        if not nbr_idxs:
            result[i] = 0.0
            continue
        nbr_vals = raw[np.array(nbr_idxs)]
        mu = nbr_vals.mean()
        sd = nbr_vals.std()
        result[i] = (raw[i] - mu) / (sd + eps)

    return result.reshape(-1, 1).astype(np.float32)


def _m1_cache_paths(dataset: str) -> tuple:
    """Return (map_file, names_file) cache paths for a dataset's M1 map.

    Each dataset gets its own cache so multiple Amazon categories can coexist:
      amazon     → amazon_m1_category_map.json   (unchanged from prior run)
      electronics → electronics_m1_category_map.json
      toys       → toys_m1_category_map.json
    """
    return (
        _DERIVED_DIR / f"{dataset}_m1_category_map.json",
        _DERIVED_DIR / f"{dataset}_m1_class_names.json",
    )


class LabelFactory:
    """
    Factory for generating labels based on task type.
    """
    
    @staticmethod
    def generate_labels(task_type: str, node_type: str, data_manager,
                       df: pd.DataFrame, graph: Optional[nx.Graph] = None,
                       node_list: Optional[List] = None):
        """
        Generate labels based on task type.
        
        Args:
            task_type: 'M1', 'M2', 'M3', 'M4', 'M5', or 'M6'
            node_type: 'N7', 'N8', or 'N9'
            data_manager: GenericDataManager instance
            df: DataFrame with sample data
            graph: NetworkX graph (required for M2/M4/M5/M6)
            node_list: List of node IDs
        
        Returns:
            Label array or (edge_list, label_array) for M3/M4
        """
        if task_type == 'M1':
            # Node categorical.
            # Datasets with 'm1_meta_url' in config use a frequency-ranked l3_cat
            # scheme derived by joining a streaming meta file (Amazon categories).
            # All other datasets use categorical_label directly from the data file.
            if data_manager.config.get('m1_meta_url'):
                return LabelFactory._generate_meta_m1(
                    node_type, df, node_list,
                    dataset=data_manager.dataset,
                    meta_url=data_manager.config['m1_meta_url'],
                    data_root=data_manager.dataset_path,
                )
            return data_manager.get_category_labels(node_type, df, node_list)
        
        elif task_type == 'M2':
            # Node scalar — optionally re-expressed as local z-score (requires graph)
            labels = data_manager.get_scalar_labels(node_type, df, graph, node_list)
            if (data_manager.config.get('m2_transform') == 'local_zscore'
                    and graph is not None):
                labels = _apply_local_zscore(labels, node_list, graph)
            return labels
        
        elif task_type == 'M3':
            # Edge categorical (same/cross category)
            return LabelFactory._generate_edge_categorical(df, graph, node_type, data_manager)
        
        elif task_type == 'M4':
            # Edge scalar (label difference)
            return LabelFactory._generate_edge_scalar(df, graph, node_type, data_manager)
        
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    @staticmethod
    def _generate_edge_categorical(df: pd.DataFrame, graph: nx.Graph,
                                   node_type: str, data_manager) -> Tuple[List, np.ndarray]:
        """
        M3: Edge categorical labels (same/cross category).
        
        Returns:
            (edge_list, labels) where labels[i] = 0 (same category) or 1 (cross category)
        """
        # Get category mapping
        if node_type == 'N7':
            node_to_cat = dict(zip(df['primary_id'], df['categorical_label']))
        
        elif node_type == 'N8':
            # For N8, compute dominant category
            sec_to_cats = defaultdict(list)
            
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                cat = row['categorical_label']
                
                if sec_id is None:
                    continue
                if isinstance(cat, float) and np.isnan(cat):
                    continue
                
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_cats[sid].append(cat)
                else:
                    sec_to_cats[sec_id].append(cat)
            
            # Compute mode category
            node_to_cat = {}
            for nid, cats in sec_to_cats.items():
                if cats:
                    node_to_cat[nid] = Counter(cats).most_common(1)[0][0]
        
        else:
            raise ValueError("M3 undefined for N9")
        
        # Label each edge
        edge_list = list(graph.edges())
        labels = []
        
        for u, v in edge_list:
            cat_u = node_to_cat.get(u)
            cat_v = node_to_cat.get(v)
            
            if cat_u is None or cat_v is None:
                labels.append(-1)  # Unknown—will be masked
            elif cat_u == cat_v:
                labels.append(0)  # Same category
            else:
                labels.append(1)  # Cross category
        
        return edge_list, np.array(labels, dtype=np.int64)
    
    @staticmethod
    def _generate_edge_scalar(df: pd.DataFrame, graph: nx.Graph,
                             node_type: str, data_manager) -> Tuple[List, np.ndarray]:
        """
        M4: Edge scalar labels (absolute difference in scalar labels).
        
        Returns:
            (edge_list, labels) where labels[i] = |scalar_u - scalar_v|
        """
        # First get node-level scalar labels
        node_list = list(graph.nodes())
        scalar_labels = data_manager.get_scalar_labels(node_type, df, graph, node_list)
        
        # Create mapping
        node_to_scalar = dict(zip(node_list, scalar_labels.flatten()))
        
        # Compute edge labels
        edge_list = list(graph.edges())
        labels = []
        
        for u, v in edge_list:
            scalar_u = node_to_scalar.get(u, 0.0)
            scalar_v = node_to_scalar.get(v, 0.0)
            
            diff = abs(scalar_u - scalar_v)
            labels.append(diff)
        
        return edge_list, np.array(labels, dtype=np.float32)
    
    # ── Meta-join M1 helpers (generic — works for any dataset with m1_meta_url) ──

    @staticmethod
    def _get_m1_category_map(dataset: str, meta_url: str, data_root: Path) -> Tuple[dict, list]:
        """Return (asin_to_idx, class_names) for a meta-join M1 subcategory scheme.

        Generic implementation: works for any dataset that has an m1_meta_url in
        its dataset yaml and stores data under data/{dataset}/{split}/raw.jsonl.

        Cache is per-dataset (e.g. amazon_m1_category_map.json for dataset='amazon'),
        so multiple Amazon categories can coexist in code/derived/ without collision.

        Algorithm:
          1. Collect all aggregate_ids from data_root/{train,test}/raw.jsonl.
          2. Stream meta_url to find each aggregate_id's level-3 category.
          3. Frequency-rank categories; keep the fewest named classes such that
             the Other bucket is ≤5% of the full population.
          4. Write per-dataset cache files; return (asin_to_idx, class_names).
        """
        map_file, names_file = _m1_cache_paths(dataset)

        if map_file.exists() and names_file.exists():
            with open(map_file) as f:
                asin_to_idx = json.load(f)
            with open(names_file) as f:
                class_names = json.load(f)
            return asin_to_idx, class_names

        import requests

        # Step 1: collect all aggregate_ids (parent_asins) from both splits
        all_agg_ids = []
        needed_asins: set = set()
        for split in ("train", "test"):
            rows = [json.loads(l) for l in (data_root / split / "raw.jsonl").open()]
            for row in rows:
                agg = row.get("aggregate_id")
                all_agg_ids.append(agg)
                if agg:
                    needed_asins.add(agg)
        total_rows = len(all_agg_ids)

        # Step 2: stream meta to collect parent_asin → level-3 category
        asin_to_l3: dict = {}
        found: set = set()
        print(f"[{dataset} M1] Streaming meta for {len(needed_asins):,} parent_asins...")
        with requests.get(meta_url, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pa = obj.get("parent_asin")
                if pa in needed_asins and pa not in found:
                    cats = obj.get("categories", [])
                    if isinstance(cats, list):
                        l3 = cats[2] if len(cats) > 2 else (cats[1] if len(cats) > 1 else None)
                    else:
                        l3 = None
                    asin_to_l3[pa] = l3
                    found.add(pa)
                if len(found) >= len(needed_asins):
                    break

        # Step 3: count l3_cat frequency across full population
        l3_counts: Counter = Counter()
        n_unmatched = 0
        for agg in all_agg_ids:
            l3 = asin_to_l3.get(agg) if agg else None
            if l3:
                l3_counts[l3] += 1
            else:
                n_unmatched += 1

        # Step 4: greedy class selection — keep adding until Other ≤5% of population
        other_cap   = total_rows * 0.05
        tail_budget = other_cap - n_unmatched  # headroom left for named-class tail
        named: list = []
        covered = 0
        for cat, cnt in l3_counts.most_common():
            named.append(cat)
            covered += cnt
            if (total_rows - covered - n_unmatched) <= tail_budget:
                break

        other_label = len(named)
        cat_to_idx  = {c: i for i, c in enumerate(named)}
        class_names = named + ["Other"]

        # Build parent_asin → class_idx (unmapped → Other)
        asin_to_idx: dict = {}
        for pa in needed_asins:
            l3 = asin_to_l3.get(pa)
            asin_to_idx[pa] = cat_to_idx.get(l3, other_label) if l3 else other_label

        # Validate and log
        other_count = total_rows - covered
        other_pct   = other_count / total_rows * 100
        majority    = named[0] if named else "N/A"
        maj_pct     = l3_counts[majority] / total_rows * 100 if named else 0.0
        compliant   = "≤5% ✓" if other_pct <= 5.0 else f"OVER CAP ✗ (floor={named[-1]}: {l3_counts[named[-1]]} rows)"
        print(f"[{dataset} M1] {len(named)} named classes + Other  (total {len(class_names)} classes)")
        print(f"[{dataset} M1] Majority: {majority} ({maj_pct:.1f}%)")
        print(f"[{dataset} M1] Other: {other_count:,} / {total_rows:,} rows ({other_pct:.1f}%) — {compliant}")

        # Write per-dataset cache
        _DERIVED_DIR.mkdir(exist_ok=True)
        with open(map_file, "w") as f:
            json.dump(asin_to_idx, f)
        with open(names_file, "w") as f:
            json.dump(class_names, f)

        return asin_to_idx, class_names

    @staticmethod
    def _generate_meta_m1(node_type: str, df: pd.DataFrame, node_list: Optional[List],
                           dataset: str, meta_url: str, data_root: Path) -> np.ndarray:
        """M1 for datasets using a meta-join frequency-ranked category scheme."""
        asin_to_idx, class_names = LabelFactory._get_m1_category_map(dataset, meta_url, data_root)
        n_classes  = len(class_names)
        other_idx  = n_classes - 1

        if node_type == 'N7':
            # One review per unique primary_id (asin); label via its parent_asin
            pid_to_agg: dict = {}
            for pid, agg in zip(df['primary_id'], df['aggregate_id']):
                if pid not in pid_to_agg:
                    pid_to_agg[pid] = agg
            if node_list is None:
                node_list = list(dict.fromkeys(df['primary_id'].tolist()))
            indices = [asin_to_idx.get(pid_to_agg.get(nid), other_idx) for nid in node_list]

        elif node_type == 'N8':
            # One user per node; label = mode subcategory across their reviews
            sec_to_indices: defaultdict = defaultdict(list)
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                idx    = asin_to_idx.get(row.get('aggregate_id'), other_idx)
                if sec_id is None:
                    continue
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_indices[sid].append(idx)
                else:
                    sec_to_indices[sec_id].append(idx)
            if node_list is None:
                node_list = sorted(sec_to_indices.keys())
            indices = [
                Counter(sec_to_indices[nid]).most_common(1)[0][0]
                if sec_to_indices.get(nid) else other_idx
                for nid in node_list
            ]

        elif node_type == 'N9':
            # One product family per node; aggregate_id IS the parent_asin
            if node_list is None:
                node_list = sorted(df['aggregate_id'].dropna().unique().tolist())
            indices = [asin_to_idx.get(nid, other_idx) for nid in node_list]

        else:
            raise ValueError(f"Unknown node_type: {node_type}")

        one_hot = np.zeros((len(node_list), n_classes), dtype=np.float32)
        for i, idx in enumerate(indices):
            if 0 <= idx < n_classes:
                one_hot[i, idx] = 1.0
        return one_hot

