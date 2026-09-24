"""
Builds a PyTorch Geometric Data object (node/edge/global task shape, with
train/val/test masks) for one (M, N, E, T) variant, by pulling node lists,
embeddings, edges, and labels from a GenericDataManager/EdgeFactory/
LabelFactory and assembling the right Data layout for the task type.

Reads:
  - Nothing directly — delegates all file I/O to the GenericDataManager
    instance passed into __init__.

Writes:
  - Nothing directly — returns an in-memory torch_geometric.data.Data object;
    callers (e.g. the N8-rebuild scripts under data/graphrag/) may cache it
    to a .pt file themselves.

Usage:
  Not run directly. Instantiated as TAGConstructor(data_manager) and called
  as tc.construct(variant, df, split) by experiment_runner.py and the
  various one-off rebuild/figure scripts.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split

from generic_data_manager import GenericDataManager
from edge_factory import EdgeFactory
from label_factory import LabelFactory


class TAGConstructor:
    """
    Constructs PyTorch Geometric Data objects for TAG variants.
    """
    
    def __init__(self, data_manager: GenericDataManager):
        """
        Initialize TAG constructor.
        
        Args:
            data_manager: GenericDataManager instance
        """
        self.data_manager = data_manager
    
    def construct(self, variant: Dict, df, split: str) -> Data:
        """
        Construct a PyG Data object for a given variant.
        
        Args:
            variant: Dictionary with keys 'M', 'N', 'E', 'T'
            df: DataFrame with sample data
            split: 'train' or 'test'
        
        Returns:
            PyTorch Geometric Data object
        """
        M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']

        # Get node list
        node_list = self.data_manager.get_node_list(N, df)

        # Get embeddings
        embeddings = self.data_manager.get_embeddings(N, T, split, node_list)

        # E11b needs a pre-built graph for degree centrality; use E10b as the base
        extra_kwargs = {'embeddings': embeddings}
        if E == 'E11b':
            base_edges = EdgeFactory.build_edges(
                'E10b', N, self.data_manager, df, node_list
            )
            base_graph = nx.Graph()
            base_graph.add_nodes_from(node_list)
            for edge in base_edges:
                if len(edge) == 3:
                    base_graph.add_edge(edge[0], edge[1], weight=edge[2])
                else:
                    base_graph.add_edge(edge[0], edge[1])
            extra_kwargs['base_graph'] = base_graph
            # k_structural: configurable via {dataset}_dataset.yaml, defaults
            # to 5 (k=50 degenerates under centrality-tie clustering; see
            # _build_structural_similarity in edge_factory.py for the evidence).
            extra_kwargs['k_structural'] = self.data_manager.config.get('k_structural', 5)

        # Build edges
        edges = EdgeFactory.build_edges(
            E, N, self.data_manager, df, node_list,
            **extra_kwargs
        )
        
        # Create NetworkX graph
        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        
        # Add edges (handle weighted and unweighted)
        for edge in edges:
            if len(edge) == 2:
                graph.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                graph.add_edge(edge[0], edge[1], weight=edge[2])
        
        # Generate labels
        if M in ['M3', 'M4']:
            # Edge-level tasks
            edge_list, edge_labels = LabelFactory.generate_labels(
                M, N, self.data_manager, df, graph, node_list
            )
            return self._build_edge_task_data(
                node_list, embeddings, edge_list, edge_labels, graph
            )
        
        elif M in ['M5', 'M6']:
            # Global tasks
            global_label = LabelFactory.generate_labels(
                M, N, self.data_manager, df, graph, node_list
            )
            return self._build_global_task_data(
                node_list, embeddings, graph, global_label
            )
        
        else:
            # Node-level tasks (M1, M2)
            labels = LabelFactory.generate_labels(
                M, N, self.data_manager, df, graph, node_list
            )
            return self._build_node_task_data(
                node_list, embeddings, labels, graph
            )
    
    def _build_node_task_data(self, node_list, embeddings: np.ndarray,
                              labels: np.ndarray, graph: nx.Graph) -> Data:
        """
        Build PyG Data for node-level tasks (M1, M2).
        """
        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        N = len(node_list)
        
        # Extract edges
        edge_list = []
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
        
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).T
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create Data object
        data = Data()
        data.num_nodes = N
        data.x = torch.tensor(embeddings, dtype=torch.float32)
        data.y = torch.tensor(labels, dtype=torch.float32)
        data.edge_index = edge_index
        
        # Create train/val/test splits
        # For categorical: nodes with any label > 0. For scalar (N,1): all nodes.
        if labels.shape[1] == 1:
            # Scalar task — exclude NaN labels (e.g. missing prices in history)
            has_target = np.isfinite(labels.flatten())
            idx = np.where(has_target)[0]
        else:
            # Categorical task — only nodes with labels
            has_target = (labels.sum(axis=1) > 0)
            idx = np.where(has_target)[0]
        
        if len(idx) < 3:
            # Not enough labeled nodes
            idx = np.arange(N)
        
        # Split: 70% train, 15% val, 15% test
        train_idx, test_idx = train_test_split(idx, train_size=0.7, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15/0.7, random_state=42)
        
        # Create masks
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(N, dtype=torch.bool)
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        return data
    
    def _build_edge_task_data(self, node_list, embeddings: np.ndarray,
                              edge_list, edge_labels: np.ndarray,
                              graph: nx.Graph) -> Data:
        """
        Build PyG Data for edge-level tasks (M3, M4).
        """
        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        N = len(node_list)
        
        # Build full edge_index for message passing
        all_edges = []
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                all_edges.append([node_to_idx[u], node_to_idx[v]])
        
        if len(all_edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(all_edges, dtype=torch.long).T
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Build target edge index (edges with known labels only).
        # valid_edges and valid_labels must stay in sync: filter on both the
        # >= 0 sentinel check (M3) AND node membership (edges to out-of-sample
        # nodes, e.g. history structural_edges referencing books not in this
        # sample's node_list) simultaneously.
        valid_edges = []
        valid_label_list = []
        for i, (u, v) in enumerate(edge_list):
            if edge_labels[i] >= 0 and u in node_to_idx and v in node_to_idx:
                valid_edges.append([node_to_idx[u], node_to_idx[v]])
                valid_label_list.append(edge_labels[i])
        valid_labels = (np.array(valid_label_list, dtype=edge_labels.dtype)
                        if valid_label_list else np.array([], dtype=edge_labels.dtype))
        
        E = len(valid_edges)
        if E == 0:
            target_edge_index = torch.empty((2, 0), dtype=torch.long)
            y_edge = torch.empty(0, dtype=torch.long if edge_labels.dtype == np.int64 else torch.float32)
        else:
            target_edge_index = torch.tensor(valid_edges, dtype=torch.long).T
            if edge_labels.dtype == np.int64:
                y_edge = torch.tensor(valid_labels, dtype=torch.long)
            else:
                y_edge = torch.tensor(valid_labels, dtype=torch.float32)
        
        # Edge-level train/val/test split
        edge_idx = np.arange(E)
        if E >= 3:
            train_eidx, test_eidx = train_test_split(edge_idx, train_size=0.7, random_state=42)
            train_eidx, val_eidx = train_test_split(train_eidx, test_size=0.15/0.7, random_state=42)
        else:
            train_eidx = val_eidx = test_eidx = edge_idx
        
        edge_train_mask = torch.zeros(E, dtype=torch.bool)
        edge_train_mask[train_eidx] = True
        edge_val_mask = torch.zeros(E, dtype=torch.bool)
        edge_val_mask[val_eidx] = True
        edge_test_mask = torch.zeros(E, dtype=torch.bool)
        edge_test_mask[test_eidx] = True
        
        # Create Data object
        data = Data()
        data.num_nodes = N
        data.x = torch.tensor(embeddings, dtype=torch.float32)
        data.edge_index = edge_index
        data.target_edge_index = target_edge_index
        data.y = y_edge
        data.train_mask = edge_train_mask
        data.val_mask = edge_val_mask
        data.test_mask = edge_test_mask
        
        return data
    
    def _build_global_task_data(self, node_list, embeddings: np.ndarray,
                                graph: nx.Graph, global_label) -> Data:
        """
        Build PyG Data for global tasks (M5, M6).
        """
        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        N = len(node_list)
        
        # Extract edges
        edge_list = []
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
        
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).T
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create Data object
        data = Data()
        data.num_nodes = N
        data.x = torch.tensor(embeddings, dtype=torch.float32)
        data.edge_index = edge_index
        
        # Global label (single value for entire graph)
        if isinstance(global_label, (int, np.integer)):
            data.y = torch.tensor([global_label], dtype=torch.long)
        else:
            data.y = torch.tensor([global_label], dtype=torch.float32)
        
        return data

