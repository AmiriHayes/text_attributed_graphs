"""
Enumerates valid (M, N, E, T) TAG variant combinations for a dataset by
reading its {dataset}_variants.yaml (allowed N/E per task, remove_list rules)
and {dataset}_dataset.yaml (e.g. has_secondary_id gating N8 availability).

Reads:
  - data/configs/{dataset}_variants.yaml
  - data/configs/{dataset}_dataset.yaml

Writes:
  - Nothing.

Usage:
  Not run directly. Instantiated by code/experiment_runner.py as
  VariantRegistry(dataset, config_path='data/configs') and iterated via
  registry.enumerate_variants().
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Iterator


class VariantRegistry:
    """
    Registry for managing valid TAG variants from YAML configuration.
    """
    
    def __init__(self, dataset: str, config_path: str = "data/configs"):
        """
        Initialize variant registry.
        
        Args:
            dataset: one of arxiv, amazon, history, electronics, toys
            config_path: Path to configs directory
        """
        self.dataset = dataset
        self.config_path = Path(config_path)
        
        # Load variant configuration
        variant_file = self.config_path / f"{dataset}_variants.yaml"
        with open(variant_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load dataset configuration for validation
        dataset_file = self.config_path / f"{dataset}_dataset.yaml"
        with open(dataset_file, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
    
    def enumerate_variants(self, text_fidelities: List[str] = None) -> Iterator[Dict]:
        """
        Enumerate all valid (M, N, E, T) combinations.
        
        Args:
            text_fidelities: List of T values to include (default: ['T12a', 'T12b', 'T12e'])
        
        Yields:
            Dictionary with keys 'M', 'N', 'E', 'T'
        """
        if text_fidelities is None:
            text_fidelities = ['T12a', 'T12b', 'T12e']
        
        variants_dict = self.config.get('variants', {})

        # M5/M6 are derived post-hoc from M1/M2 runs — not enumerated here.
        # Their rows are emitted as side-effects inside _run_one (experiment_runner.py).
        for M in ['M1', 'M2', 'M3', 'M4']:
            if M not in variants_dict:
                continue

            for ne_combo in variants_dict[M]:
                N = ne_combo['N']
                E = ne_combo['E']

                # Validate this (M, N, E) combination
                if not self._is_valid_combination(M, N, E):
                    continue

                # Enumerate text fidelities
                for T in text_fidelities:
                    yield {'M': M, 'N': N, 'E': E, 'T': T}
    
    def _is_valid_combination(self, M: str, N: str, E: str) -> bool:
        """
        Validate (M, N, E) combination against dataset constraints.
        
        Args:
            M: Task type
            N: Node type
            E: Edge type
        
        Returns:
            True if valid, False otherwise
        """
        # Check N8 availability
        if N == 'N8' and not self.dataset_config.get('has_secondary_id', False):
            return False
        
        # E10c for history requires has_structural_edges (handled by remove_list in YAML)
        # No additional gating needed here beyond what the YAML remove_list covers
        
        # Check remove list
        remove_list = self.config.get('remove_list', [])
        for rule_entry in remove_list:
            rule = rule_entry.get('rule', '')
            if self._matches_remove_rule(M, N, E, rule):
                return False
        
        return True
    
    def _matches_remove_rule(self, M: str, N: str, E: str, rule: str) -> bool:
        """
        Check if (M, N, E) matches any tuple in a remove rule string.

        A rule string may contain multiple comma-separated tuples, e.g.:
          "(M1, N7, E10a), (M1, N7, E11c), (M3, *, E11*)"
        Returns True if (M, N, E) matches ANY of them.
        """
        for match in re.finditer(r'\(([^)]+)\)', rule):
            parts = [p.strip() for p in match.group(1).split(',')]
            if len(parts) != 3:
                continue
            if (self._matches_pattern(M, parts[0]) and
                    self._matches_pattern(N, parts[1]) and
                    self._matches_pattern(E, parts[2])):
                return True
        return False
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """
        Check if value matches pattern (supports * wildcard).
        
        Args:
            value: Actual value (e.g., 'M1', 'N7', 'E10a')
            pattern: Pattern (e.g., 'M1', '*', 'E11*')
        
        Returns:
            True if matches
        """
        if pattern == '*':
            return True
        
        if pattern.endswith('*'):
            # Prefix match
            prefix = pattern[:-1]
            return value.startswith(prefix)
        
        return value == pattern
    
    def get_variant_count(self) -> Dict[str, int]:
        """
        Get count of variants per task type.
        
        Returns:
            Dictionary mapping task type to count
        """
        counts = {}
        variants_dict = self.config.get('variants', {})

        for M in ['M1', 'M2', 'M3', 'M4']:
            if M in variants_dict:
                valid_count = sum(
                    1 for ne_combo in variants_dict[M]
                    if self._is_valid_combination(M, ne_combo['N'], ne_combo['E'])
                )
                counts[M] = valid_count
            else:
                counts[M] = 0

        # M5/M6 combos mirror M1/M2 exactly (derived metrics, not independent runs)
        counts['M5'] = counts['M1']
        counts['M6'] = counts['M2']

        return counts
    
    def get_total_variant_count(self, text_fidelities: List[str] = None) -> int:
        """
        Get total number of valid variants.
        
        Args:
            text_fidelities: List of T values (default: ['T12a', 'T12b', 'T12e'])
        
        Returns:
            Total count
        """
        if text_fidelities is None:
            text_fidelities = ['T12a', 'T12b', 'T12e']
        
        return sum(1 for _ in self.enumerate_variants(text_fidelities))

