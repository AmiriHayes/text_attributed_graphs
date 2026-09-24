"""
Abstract base class defining the interface every dataset-specific data manager
(currently only GenericDataManager) must implement to load raw data, node
lists, text, embeddings, and labels for TAG construction.

Reads:
  - Nothing directly — pure abstract interface, no I/O of its own.

Writes:
  - Nothing.

Usage:
  Not run directly. Subclassed by code/generic_data_manager.py's
  GenericDataManager, which is what experiment_runner.py actually
  instantiates.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd


class BaseDataManager(ABC):
    """
    Abstract interface for dataset management.
    All dataset-specific implementations must implement these methods.
    """
    
    @abstractmethod
    def load_data(self, split: str, sample_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Load raw data for a given split.
        
        Args:
            split: 'train' or 'test'
            sample_idx: If provided, load sample_{idx:02d}.jsonl; else load raw.jsonl
        
        Returns:
            DataFrame with generic schema columns
        """
        pass
    
    @abstractmethod
    def get_node_list(self, node_type: str, df: pd.DataFrame) -> List:
        """
        Get list of node identifiers for a given node type.
        
        Args:
            node_type: 'N7' (primary), 'N8' (secondary), or 'N9' (aggregate)
            df: DataFrame to extract nodes from
        
        Returns:
            List of node identifiers
        """
        pass
    
    @abstractmethod
    def get_node_text(self, primary_id, fidelity: str, df: pd.DataFrame) -> str:
        """
        Get text for a node at specified fidelity level.
        
        Args:
            primary_id: Node identifier
            fidelity: 'T12a' (contextual), 'T12b' (standard), or 'T12e' (baseline)
            df: DataFrame containing the node
        
        Returns:
            Text string
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, node_type: str, fidelity: str, split: str,
                      node_list: Optional[List] = None) -> np.ndarray:
        """
        Get embeddings for nodes at specified fidelity level.
        
        Args:
            node_type: 'N7', 'N8', or 'N9'
            fidelity: 'T12a', 'T12b', or 'T12e'
            split: 'train' or 'test'
            node_list: Optional list to filter embeddings to specific nodes
        
        Returns:
            Embedding array (n_nodes, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_category_labels(self, node_type: str, df: pd.DataFrame,
                           node_list: Optional[List] = None) -> np.ndarray:
        """
        Get categorical labels for M1/M3 tasks.
        
        Args:
            node_type: 'N7', 'N8', or 'N9'
            df: DataFrame containing nodes
            node_list: Optional list to filter labels
        
        Returns:
            One-hot encoded labels (n_nodes, n_classes) or class indices
        """
        pass
    
    @abstractmethod
    def get_scalar_labels(self, node_type: str, df: pd.DataFrame,
                         graph=None, node_list: Optional[List] = None) -> np.ndarray:
        """
        Get scalar labels for M2/M4 tasks.
        
        Args:
            node_type: 'N7', 'N8', or 'N9'
            df: DataFrame containing nodes
            graph: Optional graph for computing centrality (ArXiv M2)
            node_list: Optional list to filter labels
        
        Returns:
            Scalar labels (n_nodes, 1)
        """
        pass
    
    @abstractmethod
    def get_structural_edges(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Get pre-given structural edges (E10d). ???
        
        Args:
            df: DataFrame containing structural_edges column
        
        Returns:
            List of (source, target) tuples
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """
        Get dataset configuration.
        
        Returns:
            Dictionary with dataset config (has_secondary_id, has_structural_edges, etc.)
        """
        pass

