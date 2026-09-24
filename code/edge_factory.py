"""
Builds the edge list for a given (edge_type, node_type) pair, dispatching
dataset-specific edge shape via config flags read from the dataset's YAML
(secondary_id_is_list, has_structural_edges, has_secondary_id) rather than
branching on dataset name. E11c is deprecated (raises NotImplementedError); E11b's k defaults to 5,
overridable via the k_structural kwarg.

Reads:
  - Nothing directly — operates on an in-memory DataFrame (df) and
    data_manager.config already loaded by the caller.

Writes:
  - Nothing.

Usage:
  Not run directly. Called by code/tag_constructor.py's
  TAGConstructor.construct() via EdgeFactory.build_edges(...).
"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


class EdgeFactory:
    """
    Factory for constructing edges based on edge type.
    Each method returns a list of (source, target, [weight]) tuples.
    """
    
    @staticmethod
    def build_edges(edge_type: str, node_type: str, data_manager,
                   df, node_list: List, **kwargs) -> List[Tuple]:
        """
        Build edges based on edge type.
        
        Args:
            edge_type: 'E10a', 'E10b', 'E10c', 'E11a', 'E11b', 'E11c'
            node_type: 'N7', 'N8', or 'N9'
            data_manager: GenericDataManager instance
            df: DataFrame with sample data
            node_list: List of node IDs for this graph
            **kwargs: Additional parameters (embeddings, threshold, etc.)
        
        Returns:
            List of (source, target) or (source, target, weight) tuples
        """
        if edge_type == 'E10a':
            return EdgeFactory._build_categorical_gt(df, node_type, node_list)
        
        elif edge_type == 'E10b':
            return EdgeFactory._build_scalar_gt(df, node_type, node_list, data_manager)
        
        elif edge_type == 'E10c':
            return EdgeFactory._build_binary_participation(df, node_type, node_list, data_manager)
        
        elif edge_type == 'E11a':
            embeddings = kwargs.get('embeddings')
            threshold = kwargs.get('threshold', 0.75)
            return EdgeFactory._build_semantic_similarity(node_list, embeddings, threshold)
        
        elif edge_type == 'E11b':
            base_graph = kwargs.get('base_graph')
            k = kwargs.get('k_structural', 5)
            return EdgeFactory._build_structural_similarity(node_list, base_graph, k=k)
        
        elif edge_type == 'E11c':
            # DEPRECATED (pre-publication audit). E11c was documented as "functional similarity based on node
            # categories" but the implementation called _build_categorical_gt --
            # the exact same function as E10a, with the same arguments. Every
            # E11c graph was byte-identical to its E10a counterpart.
            #
            # A genuinely distinct E11c (soft/hierarchical grouping one level
            # above categorical_label) was considered and rejected as
            # intractable without new data engineering:
            #   - ArXiv/History: categorical_label already IS the top-level/
            #     root category (confirmed in {dataset}_variants.yaml) -- there
            #     is no coarser level in the current per-sample schema to group
            #     by. The raw multi-level ArXiv 'categories' string and any
            #     Amazon meta category level above l3_cat are not present in
            #     the sample jsonl schema; building a hierarchy would require
            #     re-parsing upstream source data.
            #   - In practice this is moot for ArXiv/History anyway: every
            #     variants.yaml already excludes E10a/E11c from M1 for these
            #     two datasets as a direct C4 label leak, so E11c never
            #     survived as a distinct M1 variant there to begin with.
            #   - Amazon/Electronics/Toys: E11c survives only for (M1, N8) --
            #     3 variants per dataset (one per text fidelity), 9 total
            #     across the project. A coarser category level than l3_cat
            #     exists in the raw Amazon meta 'categories' list (index 0/1
            #     vs. the index 2 used for l3_cat) but extracting it requires
            #     re-streaming the same meta_url with a new cache -- a data
            #     fetch, not a code fix, and out of scope here per the audit's
            #     own fallback clause.
            #
            # Per that fallback clause: E11c is deprecated rather than left as
            # a silent alias. Raise loudly so any caller still requesting it
            # is forced to notice, instead of silently getting E10a's graph
            # under a different label.
            raise NotImplementedError(
                "E11c is deprecated (pre-publication audit FIX 1): the previous "
                "implementation was an undocumented alias for E10a (identical "
                "edges, not merely similar). It has been removed rather than "
                "kept as a silent duplicate. Remove E11c rows from "
                "{dataset}_variants.yaml before use."
            )

        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")
    
    @staticmethod
    def _build_categorical_gt(df, node_type: str, node_list: List) -> List[Tuple]:
        """
        E10a/E11c: Connect nodes with same aggregate_id.
        """
        edges = []
        
        if node_type == 'N7':
            # Group by aggregate_id
            node_to_agg = dict(zip(df['primary_id'], df['aggregate_id']))
            agg_to_nodes = defaultdict(list)
            
            for nid in node_list:
                agg = node_to_agg.get(nid)
                if agg is not None and not (isinstance(agg, float) and np.isnan(agg)):
                    agg_to_nodes[agg].append(nid)
            
            # Connect all nodes in same aggregate
            for nodes in agg_to_nodes.values():
                for u, v in combinations(nodes, 2):
                    edges.append((u, v))
        
        elif node_type == 'N8':
            # For N8, need to compute dominant aggregate_id per secondary entity
            sec_to_agg = defaultdict(list)
            
            for _, row in df.iterrows():
                sec_id = row['secondary_id']
                agg_id = row['aggregate_id']
                
                if sec_id is None:
                    continue
                if isinstance(agg_id, float) and np.isnan(agg_id):
                    continue
                
                # Handle list (arxiv) vs single value (amazon)
                if isinstance(sec_id, list):
                    for sid in sec_id:
                        sec_to_agg[sid].append(agg_id)
                else:
                    sec_to_agg[sec_id].append(agg_id)
            
            # Compute mode aggregate for each N8 node
            node_to_agg = {}
            for nid in node_list:
                agg_list = sec_to_agg.get(nid, [])
                if agg_list:
                    mode_agg = Counter(agg_list).most_common(1)[0][0]
                    node_to_agg[nid] = mode_agg
            
            # Group by dominant aggregate
            agg_to_nodes = defaultdict(list)
            for nid, agg in node_to_agg.items():
                agg_to_nodes[agg].append(nid)
            
            # Connect nodes with same dominant aggregate
            for nodes in agg_to_nodes.values():
                for u, v in combinations(nodes, 2):
                    edges.append((u, v))
        
        elif node_type == 'N9':
            # N9 nodes ARE aggregates—undefined
            raise ValueError("E10a/E11c undefined for N9 (nodes ARE categories)")
        
        return edges
    
    @staticmethod
    def _build_scalar_gt(df, node_type: str, node_list: List, data_manager) -> List[Tuple]:
        """
        E10b: Weighted co-participation edges, dispatched via config keys.
        - secondary_id_is_list=True  → coauthorship projection (e.g. ArXiv)
        - has_structural_edges=True  → structural neighbour co-occurrence (e.g. History)
        - has_secondary_id=True, secondary_id_is_list=False → scalar user-product edges (e.g. Amazon variants)
        """
        cfg = data_manager.config
        if cfg.get('secondary_id_is_list', False):
            return EdgeFactory._build_arxiv_coauthorship(df, node_type, node_list)
        elif cfg.get('has_structural_edges', False):
            return EdgeFactory._build_history_neighbour_cooccurrence(df, node_list)
        elif cfg.get('has_secondary_id', False):
            return EdgeFactory._build_amazon_scalar_gt(df, node_type, node_list)
        else:
            raise ValueError(f"E10b not defined for {data_manager.dataset}: no matching config dispatch")
    
    @staticmethod
    def _build_arxiv_coauthorship(df, node_type: str, node_list: List) -> List[Tuple]:
        """
        ArXiv E10b: Weighted by shared authors (N7) or co-authored papers (N8).
        """
        edges = defaultdict(int)
        
        if node_type == 'N7':
            # Paper-paper edges weighted by shared authors
            # Build inverted index: author -> papers
            author_to_papers = defaultdict(set)
            for _, row in df.iterrows():
                paper_id = row['primary_id']
                if paper_id not in node_list:
                    continue
                
                authors = row['secondary_id'] or []
                for author in authors:
                    author_to_papers[author].add(paper_id)
            
            # For each author, connect all their papers
            for papers in author_to_papers.values():
                for p1, p2 in combinations(papers, 2):
                    edge_key = tuple(sorted([p1, p2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N8':
            # Author-author edges weighted by co-authored papers
            # Build paper -> authors mapping
            paper_to_authors = {}
            for _, row in df.iterrows():
                authors = row['secondary_id'] or []
                # Filter to authors in node_list
                authors_in_graph = [a for a in authors if a in node_list]
                if len(authors_in_graph) >= 2:
                    paper_to_authors[row['primary_id']] = authors_in_graph
            
            # For each paper, connect all co-authors
            for authors in paper_to_authors.values():
                for a1, a2 in combinations(authors, 2):
                    edge_key = tuple(sorted([a1, a2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N9':
            # Category-category edges: aggregate from N7 level
            # Count cross-category shared-authorship links
            # First build N7-level edges
            n7_edges = EdgeFactory._build_arxiv_coauthorship(df, 'N7', df['primary_id'].tolist())
            
            # Map papers to categories
            paper_to_cat = dict(zip(df['primary_id'], df['aggregate_id']))
            
            # Aggregate to N9
            for u, v, weight in n7_edges:
                cat_u = paper_to_cat.get(u)
                cat_v = paper_to_cat.get(v)
                
                if cat_u and cat_v and cat_u != cat_v and cat_u in node_list and cat_v in node_list:
                    edge_key = tuple(sorted([cat_u, cat_v]))
                    edges[edge_key] += weight
        
        return [(u, v, w) for (u, v), w in edges.items()]
    
    @staticmethod
    def _build_amazon_scalar_gt(df, node_type: str, node_list: List) -> List[Tuple]:
        """
        Amazon E10b: Product-product by shared users, user-user by shared products.
        """
        edges = defaultdict(int)
        
        if node_type == 'N7':
            # Product-product edges weighted by shared users
            user_to_products = defaultdict(set)
            for _, row in df.iterrows():
                product_id = row['primary_id']
                user_id = row['secondary_id']
                
                if product_id not in node_list or user_id is None:
                    continue
                
                user_to_products[user_id].add(product_id)
            
            # Connect products reviewed by same user
            for products in user_to_products.values():
                for p1, p2 in combinations(products, 2):
                    edge_key = tuple(sorted([p1, p2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N8':
            # User-user edges weighted by reviewed same products
            product_to_users = defaultdict(set)
            for _, row in df.iterrows():
                user_id = row['secondary_id']
                product_id = row['primary_id']
                
                if user_id is None or user_id not in node_list:
                    continue
                
                product_to_users[product_id].add(user_id)
            
            # Connect users who reviewed same product
            for users in product_to_users.values():
                for u1, u2 in combinations(users, 2):
                    edge_key = tuple(sorted([u1, u2]))
                    edges[edge_key] += 1
        
        elif node_type == 'N9':
            # Family-family edges: aggregate from N7 level
            n7_edges = EdgeFactory._build_amazon_scalar_gt(df, 'N7', df['primary_id'].tolist())
            
            product_to_family = dict(zip(df['primary_id'], df['aggregate_id']))
            
            # Aggregate to N9
            for u, v, weight in n7_edges:
                fam_u = product_to_family.get(u)
                fam_v = product_to_family.get(v)
                
                if fam_u and fam_v and fam_u != fam_v and fam_u in node_list and fam_v in node_list:
                    edge_key = tuple(sorted([fam_u, fam_v]))
                    edges[edge_key] += weight
        
        return [(u, v, w) for (u, v), w in edges.items()]
    
    @staticmethod
    def _build_history_neighbour_cooccurrence(df, node_list: List) -> List[Tuple]:
        """
        History E10b: Weighted by shared entries in structural_edges lists.
        """
        # Build node -> neighbours mapping
        node_to_neighbours = {}
        for _, row in df.iterrows():
            nid = row['primary_id']
            if nid in node_list:
                neighbours = set(row['structural_edges'] or [])
                node_to_neighbours[nid] = neighbours
        
        # Count shared neighbours for each pair
        edges = defaultdict(int)
        for n1, n2 in combinations(node_list, 2):
            neighbours1 = node_to_neighbours.get(n1, set())
            neighbours2 = node_to_neighbours.get(n2, set())
            
            shared = len(neighbours1 & neighbours2)
            if shared > 0:
                edges[(n1, n2)] = shared
        
        return [(u, v, w) for (u, v), w in edges.items()]
    
    @staticmethod
    def _build_binary_participation(df, node_type: str, node_list: List, data_manager) -> List[Tuple]:
        """
        E10c: Binary (0/1) N-N participation edges projected from the dataset's
        pre-given relational structure.

        ArXiv  N7: paper-paper if they share ≥1 author
               N8: author-author if they share ≥1 paper
               N9: category-category if any cross-category paper pair shares an author
        Amazon N7: product-product if reviewed by ≥1 same user
               N8: user-user if they reviewed ≥1 same product
               N9: family-family if any cross-family product pair shares a reviewer
        History N7: structural_edges column (already N-N)
        """
        cfg = data_manager.config
        node_set = set(node_list)

        if cfg.get('has_structural_edges', False):
            return data_manager.get_structural_edges(df)

        if cfg.get('secondary_id_is_list', False):
            if node_type == 'N7':
                # paper-paper: share ≥1 author
                author_to_papers = defaultdict(set)
                for _, row in df.iterrows():
                    pid = row['primary_id']
                    if pid not in node_set:
                        continue
                    for a in (row['secondary_id'] or []):
                        author_to_papers[a].add(pid)
                seen = set()
                edges = []
                for papers in author_to_papers.values():
                    for p1, p2 in combinations(papers, 2):
                        key = (min(p1, p2), max(p1, p2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N8':
                # author-author: share ≥1 paper
                paper_to_authors = defaultdict(set)
                for _, row in df.iterrows():
                    for a in (row['secondary_id'] or []):
                        if a in node_set:
                            paper_to_authors[row['primary_id']].add(a)
                seen = set()
                edges = []
                for authors in paper_to_authors.values():
                    for a1, a2 in combinations(authors, 2):
                        key = (min(a1, a2), max(a1, a2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N9':
                # category-category: any cross-cat paper pair shares an author
                paper_to_cat = dict(zip(df['primary_id'], df['aggregate_id']))
                author_to_cats = defaultdict(set)
                for _, row in df.iterrows():
                    cat = paper_to_cat.get(row['primary_id'])
                    if cat not in node_set:
                        continue
                    for a in (row['secondary_id'] or []):
                        author_to_cats[a].add(cat)
                seen = set()
                edges = []
                for cats in author_to_cats.values():
                    cats_in = [c for c in cats if c in node_set]
                    for c1, c2 in combinations(cats_in, 2):
                        key = (min(c1, c2), max(c1, c2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

        if cfg.get('has_secondary_id', False) and not cfg.get('secondary_id_is_list', False):
            if node_type == 'N7':
                # product-product: reviewed by ≥1 same user
                user_to_products = defaultdict(set)
                for _, row in df.iterrows():
                    pid = row['primary_id']
                    uid = row['secondary_id']
                    if pid in node_set and uid is not None:
                        user_to_products[uid].add(pid)
                seen = set()
                edges = []
                for products in user_to_products.values():
                    for p1, p2 in combinations(products, 2):
                        key = (min(p1, p2), max(p1, p2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N8':
                # user-user: reviewed ≥1 same product
                product_to_users = defaultdict(set)
                for _, row in df.iterrows():
                    uid = row['secondary_id']
                    if uid in node_set:
                        product_to_users[row['primary_id']].add(uid)
                seen = set()
                edges = []
                for users in product_to_users.values():
                    for u1, u2 in combinations(users, 2):
                        key = (min(u1, u2), max(u1, u2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

            elif node_type == 'N9':
                # family-family: cross-family products share a reviewer
                product_to_fam = dict(zip(df['primary_id'], df['aggregate_id']))
                user_to_fams = defaultdict(set)
                for _, row in df.iterrows():
                    uid = row['secondary_id']
                    fam = product_to_fam.get(row['primary_id'])
                    if uid is not None and fam in node_set:
                        user_to_fams[uid].add(fam)
                seen = set()
                edges = []
                for fams in user_to_fams.values():
                    fams_in = [f for f in fams if f in node_set]
                    for f1, f2 in combinations(fams_in, 2):
                        key = (min(f1, f2), max(f1, f2))
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                return edges

        raise ValueError(f"E10c not defined for {data_manager.dataset}: no matching config dispatch")
    
    @staticmethod
    def _build_semantic_similarity(node_list: List, embeddings: np.ndarray,
                                   threshold: float = 0.75, chunk_size: int = 1000) -> List[Tuple]:
        """
        E11a: Semantic similarity edges based on cosine similarity.

        Computed in row-chunks rather than materializing the full N x N
        dense matrix — mathematically identical to the dense version (same
        cosine threshold, same strict-upper-triangle semantics, no
        approximation), just memory-bounded. At N=50k the full dense
        float64 matrix would be ~20GB; chunked at chunk_size=1000, peak
        memory per chunk is ~1000 x N x 8 bytes (~400MB at N=50k).
        """
        # Replace NaN/inf and guard zero-norm rows
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float64)

        n = embeddings.shape[0]
        edges = []
        # For unit-norm vectors cosine similarity == dot product.
        # np.errstate suppresses spurious BLAS FP-exception flags.
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = embeddings[start:end]        # (rows, dim)
                rest = embeddings[start:]            # (n-start, dim) — only columns
                                                      # j>=start can ever satisfy j>i for i>=start
                sim_chunk = chunk @ rest.T            # (rows, n-start)

                rows, cols = end - start, n - start
                # global j > global i  <=>  local_j > local_i (both offset by `start`)
                upper_mask = np.arange(cols)[None, :] > np.arange(rows)[:, None]
                keep = (sim_chunk >= threshold) & upper_mask

                local_i, local_j = np.where(keep)
                for li, lj in zip(local_i, local_j):
                    gi, gj = start + li, start + lj
                    edges.append((node_list[gi], node_list[gj], float(sim_chunk[li, lj])))
        return edges

    @staticmethod
    def _build_structural_similarity(node_list: List, base_graph: nx.Graph,
                                     k: int = 5, chunk_size: int = 1000) -> List[Tuple]:
        """
        E11b: Structural similarity based on degree centrality.
        Each node is connected to its k most centrality-similar neighbors.

        DEGENERATE CONDITION (pre-publication audit): the previous default, k=50,
        degenerates into a near-complete graph whenever centrality values
        cluster tightly -- which is common, not an edge case. Centrality is
        computed over E10b's ground-truth graph, and a large fraction of
        nodes routinely tie at centrality=0 (no E10b edges at all): measured
        37.5%-100% of nodes tied across the datasets checked (e.g. ArXiv N7
        56.3% at pooled scale / 91.2% at single-sample training scale;
        Amazon N8 100%). When most nodes are tied, "k nearest by centrality
        difference" has no real selection criterion left -- it picks k
        arbitrary nodes from the tied pool, and since every tied node does
        this independently, the union of edges balloons. At k=50 this
        produced graphs 100-650x denser than E10b/E11a on the same node set
        (e.g. ArXiv N7: 447,618 edges vs. E10b's 4,021 and E11a's 692).
        Critically, this floor cannot be tuned away with a smaller k --
        even k=1 (the sparsest a k-NN construction can be) still produces
        roughly one edge per node when ties are this common, which is
        already several times denser than the ground-truth/similarity edge
        types.

        EMPIRICALLY VALIDATED FIX: k=5 (ArXiv, N7, T12a, pooled 9,178-node
        graph). Edge count: 447,618 (k=50) -> 45,686 (k=5), a 90% reduction.
        Validation experiment:
          - raw_gnn (GraphSAGE, M1, n=10 samples): k=50 mean=55.79 (std
            2.62) vs. k=5 mean=54.34 (std 3.42). Welch's t-test t=-1.065,
            p=0.302 (NOT significant), Cohen's d=-0.476. Sparsification
            does not measurably change GNN node-classification performance.
          - RAGAS composite (text-based GraphRAG, n=20 questions): k=50
            mean=0.359 vs. k=5 mean=0.639 (+0.28 absolute). This moves E11b
            from a catastrophic outlier (roughly half of every other edge
            type's score) to just below the bottom of the normal range
            (0.639 vs. 0.687-0.719 for E10b/E10c/E11a on the same node
            type).
          - Interpretation: raw_gnn is blind to this construction failure
            mode; RAGAS/retrieval quality is highly sensitive to it. This
            is one dataset/node-type/text-fidelity combination -- treat as
            a validated pilot, not a five-dataset confirmation, until
            replicated elsewhere (Amazon N8 is the natural next check,
            given its centrality ties were the most severe observed, 100%).

        k is configurable via the `k_structural` kwarg (see
        EdgeFactory.build_edges) so future users/datasets can retune it;
        defaults to 5 here to match the validated fix.

        Computed in row-chunks rather than materializing the full N x N
        dense diff matrix — same brute-force per-row k-nearest-by-
        centrality-difference result as the dense version (argpartition is
        still applied against the FULL centrality array for each row, only
        the row axis is chunked), just memory-bounded. At N=50k the full
        dense float64 diff matrix would be ~20GB; chunked at
        chunk_size=1000, peak memory per chunk is ~1000 x N x 8 bytes
        (~400MB at N=50k).
        """
        degree_cent = nx.degree_centrality(base_graph)
        cents = np.array([degree_cent.get(n, 0.0) for n in node_list], dtype=np.float64)
        n = len(node_list)
        k = min(k, n - 1)

        seen = set()
        edges = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_cents = cents[start:end]                                # (rows,)
            diff_chunk = np.abs(chunk_cents[:, None] - cents[None, :])    # (rows, n)

            local_rows = np.arange(end - start)
            diff_chunk[local_rows, start + local_rows] = np.inf           # exclude self

            knn_chunk = np.argpartition(diff_chunk, k, axis=1)[:, :k]     # (rows, k)

            for li, row_knn in enumerate(knn_chunk):
                i = start + li
                for j in row_knn:
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        sim = float(max(0.0, 1.0 - diff_chunk[li, j]))
                        edges.append((node_list[i], node_list[j], sim))
        return edges

