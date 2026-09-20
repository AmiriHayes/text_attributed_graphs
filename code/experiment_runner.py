"""
Outer training loop: for each dataset, enumerates every valid (M, N, E, T)
variant via VariantRegistry, builds a graph per split/sample via
TAGConstructor, trains a GNN (+ a cached MLP baseline) per graph via
GNNTrainer, and appends one result row per (variant, split, sample) to that
dataset's construction_performance_table CSV — resumable via the existing
(Task/Node/Edge/Text/split/sample_idx) keys already in the CSV.

Reads:
  - data/configs/{dataset}_dataset.yaml, data/configs/{dataset}_variants.yaml
    (via GenericDataManager / VariantRegistry)
  - data/{dataset}/{train,test}/samples/sample_{idx:02d}.jsonl and
    embeddings/ (via GenericDataManager — see generic_data_manager.py)

Writes:
  - {output_path}/construction_performance_table_{dataset}.csv
  - {output_path}/experiment_runner.log

Usage (from the code/ directory or after adding code/ to sys.path):
    from experiment_runner import ExperimentRunner
    runner = ExperimentRunner(base_path='data', output_path='output')
    runner.run()

Or run directly:
    python experiment_runner.py --datasets arxiv amazon --output_path output/my_run
"""

import sys
import os
import math
import time
import traceback
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# NOTE on sample counts:
# Default 10/10 kept for backward compat. Pass --n_train_samples / --n_test_samples
# to override. run_final uses 75/75 (samples 00-74), i.e. the 150 subsets the
# paper reports; samples 50-74 come from code/helpers/extend_samples.py.
# ---------------------------------------------------------------------------

N_TRAIN_SAMPLES = 10   # default; override via CLI --n_train_samples
N_TEST_SAMPLES  = 10   # default; override via CLI --n_test_samples

DATASETS = ['arxiv', 'amazon', 'history', 'electronics', 'toys']

# Map M-code to trainer task_type string
M_TO_TASK_TYPE = {
    'M1': 'categorical',
    'M2': 'scalar',
    'M3': 'edge_categorical',
    'M4': 'edge_scalar',
    'M5': 'global_categorical',   # uses GlobalTrainer (global_trainer.py)
    'M6': 'global_scalar',        # uses GlobalTrainer (global_trainer.py)
}

# Default model / trainer hyper-parameters
DEFAULT_HIDDEN_DIM = 256
DEFAULT_DROPOUT    = 0.5
DEFAULT_LR         = 0.001
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS     = 100
DEFAULT_PATIENCE   = 30

# Graphs with more than this many undirected edges are skipped: training on
# near-complete graphs is both intractable on CPU and produces degenerate
# labels (e.g. all-identical centrality → zero-variance → R2 = -∞).
MAX_EDGES = 200_000


def _setup_logging(output_path: Path) -> logging.Logger:
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('experiment_runner')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # encoding='utf-8' fixes UnicodeEncodeError on Windows: FileHandler
        # defaults to locale.getpreferredencoding() (cp1252 here), which
        # can't encode box-drawing characters used in some log messages
        # (e.g. "  └─ M5 aggregate_gap=..."). PYTHONIOENCODING
        # does not affect FileHandler's file-open encoding, only std
        # streams, so it didn't help despite being set at launch.
        fh = logging.FileHandler(output_path / 'experiment_runner.log', encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def _dense_graph_row(variant: dict, run_split: str, sample_idx: int,
                     n_edges: int, task_type: str) -> dict:
    """Return a pre-filled degenerate row for graphs that exceed MAX_EDGES."""
    nan = float('nan')
    M = variant['M']
    return {
        'Task_Idx': M, 'Node_Idx': variant['N'],
        'Edge_Idx': variant['E'], 'Text_Idx': variant['T'],
        'task_type': task_type,
        'KL': nan, 'Top1': nan, 'Top3': nan, 'Cosine': nan,
        'Accuracy': nan, 'F1': nan,
        'MAE': nan, 'MSE': nan, 'R2': nan,
        'Primary_Metric': 'final_score', 'Primary_Value': nan,
        'Performance_Band': nan,
        'S_GNN_step1': nan, 'S_MLP_step1': nan, 'Final_Score': nan,
        'normalized_score': nan,
        'number_of_nodes': nan, 'avg_text_length': nan,
        'text_vocab_entropy': nan, 'label_balance_entropy': nan,
        'output_dimension': nan,
        'run_split': run_split, 'degenerate': True,
        'nan_reason': 'dense_graph',
        'Predicted_Value': nan, 'True_Value': nan,
        'sample_idx': sample_idx,
    }


def _load_existing_keys(csv_path: Path) -> set:
    """Return set of (Task_Idx, Node_Idx, Edge_Idx, Text_Idx, run_split, sample_idx) already done."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[
            'Task_Idx', 'Node_Idx', 'Edge_Idx', 'Text_Idx', 'run_split', 'sample_idx'
        ])
        return set(zip(df['Task_Idx'], df['Node_Idx'], df['Edge_Idx'],
                       df['Text_Idx'], df['run_split'], df['sample_idx']))
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Outer loop: datasets → variants → splits → samples → train → collect row.
    """

    def __init__(
        self,
        base_path: str = 'data',
        output_path: str = 'output',
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        epochs: int = DEFAULT_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        early_stopping: bool = False,
        verbose: bool = False,
        device: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        n_train_samples: int = N_TRAIN_SAMPLES,
        n_test_samples: int = N_TEST_SAMPLES,
        record_wall_time: bool = False,
        epoch_log_variants: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ):
        self.base_path   = Path(base_path)
        self.output_path = Path(output_path)
        self.hidden_dim  = hidden_dim
        self.dropout     = dropout
        self.lr          = lr
        self.weight_decay = weight_decay
        self.epochs      = epochs
        self.patience    = patience
        self.early_stopping = early_stopping
        self.verbose     = verbose
        self.datasets    = datasets or DATASETS
        self.n_train_samples = n_train_samples
        self.n_test_samples  = n_test_samples

        # Optional, CLI-gated infrastructure additions -- OFF by default, so
        # default behavior (per-sample immediate CSV writes, no epoch
        # logging) is byte-for-byte unchanged from before these existed.
        # Both removable without touching any other code path.
        self.record_wall_time  = record_wall_time
        self.epoch_log_variants = set(epoch_log_variants or [])
        # Optional M-task filter (e.g. ['M1']), temporary/removable, gated
        # by --tasks -- unset (None) = every task type, unchanged default.
        self.tasks = set(tasks) if tasks else None

        import torch
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.logger = _setup_logging(self.output_path)

    def run(self):
        """Run all experiments."""
        for dataset in self.datasets:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Dataset: {dataset}")
            self.logger.info(f"{'='*60}")
            try:
                self._run_dataset(dataset)
            except Exception as e:
                self.logger.error(f"Fatal error for dataset '{dataset}': {e}")
                self.logger.error(traceback.format_exc())

    def _run_dataset(self, dataset: str):
        from generic_data_manager import GenericDataManager
        from tag_constructor import TAGConstructor
        from variant_registry import VariantRegistry
        from models import ModelFactory, MLPModelFactory
        from trainer import (GNNTrainer, build_result_row,
                             compute_aggregate_gap, build_derived_global_row,
                             compute_step1_score, compute_final_score,
                             compute_final_score_algebraic,
                             compute_m5_step1_score, compute_m6_step1_score)
        from global_trainer import GlobalTrainer, build_result_row_global  # deprecated — see global_trainer.py

        dm = GenericDataManager(dataset, base_path=str(self.base_path))
        builder = TAGConstructor(dm)
        registry = VariantRegistry(dataset, config_path=str(self.base_path / 'configs'))
        # MLP baseline cache: (task_type, N, T, sample_idx, split) → (s_mlp_step1, mlp_test_metrics)
        # MLP ignores edge_index, so one training run covers all E variants sharing (N, T).
        mlp_cache: dict = {}

        csv_path = self.output_path / f'construction_performance_table_{dataset}.csv'
        existing_keys = _load_existing_keys(csv_path)
        self.logger.info(f"Resuming from {len(existing_keys)} existing rows in {csv_path}")

        all_variants = list(registry.enumerate_variants())
        if self.tasks:
            all_variants = [v for v in all_variants if v['M'] in self.tasks]
        n_variants   = len(all_variants)
        rows_written  = 0

        for v_idx, variant in enumerate(all_variants):
            M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']
            variant_str = f"{M}/{N}/{E}/{T}"
            task_type   = M_TO_TASK_TYPE[M]

            self.logger.info(f"[{v_idx+1}/{n_variants}] {dataset} | {variant_str} | task={task_type}")

            # Skip variants whose node/edge/task axis was flagged degenerate
            # by characterize_dataset.py's automatic dataset characterization
            # (data/configs/{dataset}_dataset.yaml -> dataset_characterization
            # -> recommended_exclusions, e.g. 'SKIP_N8', 'SKIP_N9', 'SKIP_M1').
            # Absent block / empty list = no exclusions, nothing skipped.
            recommended_exclusions = dm.config.get('dataset_characterization', {}) \
                                               .get('recommended_exclusions', [])
            axis_exclusion_flags = {f'SKIP_{M}', f'SKIP_{N}', f'SKIP_{E}', f'SKIP_{T}'}
            hit = axis_exclusion_flags & set(recommended_exclusions)
            if hit:
                self.logger.info(f"  SKIP {variant_str}: flagged by dataset_characterization "
                                  f"({', '.join(sorted(hit))})")
                continue

            # ----------------------------------------------------------------
            # M5 / M6 — DEPRECATED global graph-level training path
            # M5/M6 are now derived post-hoc from M1/M2 runs (see trainer.py
            # compute_aggregate_gap / build_derived_global_row). The registry
            # no longer enumerates M5/M6 variants, so this block is dead code.
            # Kept for reference; GlobalTrainer is marked deprecated in global_trainer.py.
            # ----------------------------------------------------------------
            if M in ('M5', 'M6'):  # pragma: no cover
                out_dim = 2 if M == 'M5' else 1

                try:
                    # Probe first graph for edge density before committing to
                    # building and training on all 40. Dense graphs (e.g.
                    # N8/E10a category-membership cliques) take hours and
                    # produce degenerate labels anyway.
                    probe = builder.construct(variant, dm.load_data('train', 0), 'train')
                    probe_edges = probe.edge_index.shape[1] // 2
                    if probe_edges > MAX_EDGES:
                        self.logger.warning(
                            f"  SKIP {variant_str}: probe graph has {probe_edges:,} edges "
                            f"(>{MAX_EDGES:,}) — writing degenerate rows"
                        )
                        all_keys = (
                            [('train', i) for i in range(self.n_train_samples)] +
                            [('test',  i) for i in range(self.n_test_samples)]
                        )
                        for split_name, sample_idx in all_keys:
                            resume_key = (M, N, E, T, split_name, sample_idx)
                            if resume_key in existing_keys:
                                continue
                            row = _dense_graph_row(variant, split_name, sample_idx,
                                                   probe_edges, task_type)
                            out_df = pd.DataFrame([row])
                            write_header = not csv_path.exists()
                            out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                            existing_keys.add(resume_key)
                            rows_written += 1
                        continue

                    # Load all train-pool and test-pool graphs
                    train_graphs = [probe] + [
                        builder.construct(variant, dm.load_data('train', i), 'train')
                        for i in range(1, N_TRAIN_SAMPLES)
                    ]
                    test_graphs = [
                        builder.construct(variant, dm.load_data('test', i), 'test')
                        for i in range(N_TEST_SAMPLES)
                    ]

                    in_dim = train_graphs[0].x.shape[1]
                    model  = ModelFactory.create(
                        task_type=task_type,
                        in_dim=in_dim,
                        hidden_dim=self.hidden_dim,
                        out_dim=out_dim,
                        dropout=self.dropout,
                    )
                    global_trainer = GlobalTrainer(
                        model=model,
                        task_type=task_type,
                        device=self.device,
                        lr=self.lr,
                        weight_decay=self.weight_decay,
                        epochs=self.epochs,
                        patience=self.patience,
                    )
                    global_trainer.train(train_graphs)
                    self.logger.info(f"  Trained on {len(train_graphs)} train graphs")

                except Exception as e:
                    self.logger.error(f"  ERROR training {variant_str}: {e}")
                    self.logger.debug(traceback.format_exc())
                    continue

                # Evaluate on all 40 graphs individually
                for split_name, graphs in [('train', train_graphs), ('test', test_graphs)]:
                    n_samples = len(graphs)
                    for sample_idx in range(n_samples):
                        resume_key = (M, N, E, T, split_name, sample_idx)
                        if resume_key in existing_keys:
                            continue
                        try:
                            graph      = graphs[sample_idx]
                            df         = dm.load_data(split_name, sample_idx)
                            graph_result = global_trainer.evaluate_single(graph)
                            row = build_result_row_global(
                                variant=variant,
                                graph_result=graph_result,
                                graph=graph,
                                df=df,
                                trainer=global_trainer,
                                run_split=split_name,
                                sample_idx=sample_idx,
                                output_dimension=out_dim,
                            )
                            row['sample_idx'] = sample_idx

                            out_df = pd.DataFrame([row])
                            write_header = not csv_path.exists()
                            out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                            existing_keys.add(resume_key)
                            rows_written += 1

                            pv = row['Primary_Value']
                            pv_str = f"{pv:.4f}" if not (isinstance(pv, float) and math.isnan(pv)) else 'nan'
                            self.logger.info(
                                f"  {split_name} sample_{sample_idx:02d} | "
                                f"primary={row['Primary_Metric']}={pv_str} | "
                                f"degen={row['degenerate']}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"  ERROR {variant_str} {split_name} sample_{sample_idx:02d}: {e}"
                            )
                            self.logger.debug(traceback.format_exc())

                continue   # skip the M1-M4 loop below

            # ----------------------------------------------------------------
            # M1-M4 — existing node/edge training path
            # ----------------------------------------------------------------
            # record_wall_time buffers this variant's rows and writes them
            # together at the end so one wall_time_seconds total (all
            # train+test samples) can be attached to every row -- this is
            # the only behavior change the flag causes; per-sample immediate
            # writes (and therefore intra-variant resume) are only affected
            # when the flag is explicitly on. Default (flag off): identical
            # to before this existed.
            variant_t0 = time.perf_counter() if self.record_wall_time else None
            variant_row_buffer: list = []

            epoch_log_path = None
            if variant_str in self.epoch_log_variants:
                epoch_log_dir = self.output_path / 'epoch_logs'
                epoch_log_dir.mkdir(parents=True, exist_ok=True)

            for split in ['train', 'test']:
                n_samples = self.n_train_samples if split == 'train' else self.n_test_samples

                for sample_idx in range(n_samples):
                    # Resume check
                    resume_key = (M, N, E, T, split, sample_idx)
                    if resume_key in existing_keys:
                        continue

                    if variant_str in self.epoch_log_variants:
                        safe_variant = variant_str.replace('/', '_')
                        epoch_log_path = str(self.output_path / 'epoch_logs' /
                                              f'{safe_variant}_{split}{sample_idx:02d}.csv')

                    try:
                        result = self._run_one(
                            dm=dm,
                            builder=builder,
                            variant=variant,
                            task_type=task_type,
                            split=split,
                            sample_idx=sample_idx,
                            ModelFactory=ModelFactory,
                            MLPModelFactory=MLPModelFactory,
                            GNNTrainer=GNNTrainer,
                            build_result_row=build_result_row,
                            mlp_cache=mlp_cache,
                            compute_step1_score=compute_step1_score,
                            compute_final_score=compute_final_score,
                            compute_final_score_algebraic=compute_final_score_algebraic,
                            compute_m5_step1_score=compute_m5_step1_score,
                            compute_m6_step1_score=compute_m6_step1_score,
                            epoch_log_path=epoch_log_path,
                            experiment_id=f'{variant_str}_{split}{sample_idx:02d}',
                        )
                        if result is None:
                            continue

                        main_row, derived_row = result
                        main_row['sample_idx'] = sample_idx

                        rows_to_write = [main_row]
                        if derived_row is not None:
                            derived_row['sample_idx'] = sample_idx
                            rows_to_write.append(derived_row)

                        if self.record_wall_time:
                            # buffered -- written together once the whole
                            # variant (all splits/samples) finishes below
                            variant_row_buffer.extend(rows_to_write)
                            existing_keys.add(resume_key)
                            rows_written += len(rows_to_write)
                        else:
                            # unchanged default path: write main + derived
                            # rows atomically (same to_csv call) as before
                            out_df = pd.DataFrame(rows_to_write)
                            write_header = not csv_path.exists()
                            out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                            existing_keys.add(resume_key)
                            rows_written += len(rows_to_write)

                        pv = main_row['Primary_Value']
                        ns = main_row['normalized_score']
                        pv_str = f"{pv:.4f}" if isinstance(pv, float) and not math.isnan(pv) else 'nan'
                        ns_str = f"{ns:.1f}" if isinstance(ns, float) and not math.isnan(ns) else 'nan'
                        self.logger.info(
                            f"  {split} sample_{sample_idx:02d} | "
                            f"primary={main_row['Primary_Metric']}={pv_str} | "
                            f"norm={ns_str} | "
                            f"degen={main_row['degenerate']}"
                        )
                        if derived_row is not None:
                            gap = derived_row['Primary_Value']
                            gap_str = f"{gap:.4f}" if not math.isnan(gap) else 'nan'
                            self.logger.info(
                                f"  └─ {'M5' if M=='M1' else 'M6'} aggregate_gap={gap_str}"
                            )

                    except Exception as e:
                        self.logger.error(
                            f"  ERROR {variant_str} {split} sample_{sample_idx:02d}: {e}"
                        )
                        self.logger.debug(traceback.format_exc())
                        continue

            # record_wall_time: flush this variant's buffered rows now that
            # every train+test sample is done, with a single wall_time_seconds
            # total (perf_counter delta since the variant started) attached
            # to all of them.
            if self.record_wall_time and variant_row_buffer:
                wall_time = time.perf_counter() - variant_t0
                for row in variant_row_buffer:
                    row['wall_time_seconds'] = round(wall_time, 3)
                out_df = pd.DataFrame(variant_row_buffer)
                write_header = not csv_path.exists()
                out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
                self.logger.info(f"  [{variant_str}] wall_time_seconds={wall_time:.2f} "
                                  f"({len(variant_row_buffer)} rows)")

        self.logger.info(f"Dataset '{dataset}' complete. Rows written this run: {rows_written}")
        self.logger.info(f"Output: {csv_path}")

    def _run_one(
        self, dm, builder, variant, task_type, split, sample_idx,
        ModelFactory, MLPModelFactory, GNNTrainer, build_result_row, mlp_cache,
        compute_step1_score, compute_final_score, compute_final_score_algebraic,
        compute_m5_step1_score, compute_m6_step1_score,
        epoch_log_path=None, experiment_id=None,
    ):
        """Train GNN + MLP baseline for one (variant, split, sample_idx).

        Returns:
            (main_row, derived_row) where derived_row is the M5/M6 row for
            M1/M2 variants, or None for M3/M4.
            Returns None on unrecoverable error (caller must handle).
        """
        from trainer import compute_aggregate_gap, build_derived_global_row

        nan = float('nan')
        M, N, E, T = variant['M'], variant['N'], variant['E'], variant['T']

        df   = dm.load_data(split, sample_idx)
        data = builder.construct(variant, df, split)

        n_edges = data.edge_index.shape[1] // 2
        if n_edges > MAX_EDGES:
            main_row = _dense_graph_row(variant, split, sample_idx, n_edges, task_type)
            main_row['Predicted_Value'] = nan
            main_row['True_Value']      = nan
            if M in ('M1', 'M2'):
                out_dim = data.y.shape[-1] if M == 'M1' else 1
                derived_row = build_derived_global_row(
                    variant, nan, split, sample_idx, data, df, out_dim,
                    s_gnn_step1=nan, s_mlp_step1=nan, final_score=nan,
                )
                return main_row, derived_row
            return main_row, None

        if M == 'M1':
            output_dimension = data.y.shape[-1]
        elif M == 'M2':
            output_dimension = 1
        elif M == 'M3':
            output_dimension = 2
        elif M == 'M4':
            output_dimension = 1
        else:
            raise ValueError(f"Unexpected M: {M}")

        in_dim = data.x.shape[1]

        # ── GNN run ──────────────────────────────────────────────────────────
        gnn_model = ModelFactory.create(
            task_type=task_type, in_dim=in_dim,
            hidden_dim=self.hidden_dim, out_dim=output_dimension,
            dropout=self.dropout,
        )
        gnn_trainer = GNNTrainer(
            model=gnn_model, device=self.device,
            lr=self.lr, weight_decay=self.weight_decay,
            epochs=self.epochs, patience=self.patience,
            task_type=task_type,
        )
        gnn_metrics = gnn_trainer.train(
            data=data, num_classes=output_dimension,
            verbose=self.verbose, early_stopping=self.early_stopping,
            epoch_log_path=epoch_log_path, experiment_id=experiment_id,
        )
        s_gnn, r2_for_row = compute_step1_score(gnn_metrics, task_type)

        # ── MLP baseline (cached per (task_type, N, T, sample_idx, split)) ──
        # MLP ignores edge_index, so one training run serves all E variants
        # that share the same (N, T, sample_idx, split).
        # Exception: local_zscore M2 targets depend on the edge construction
        # (neighbourhood is defined per-E), so each E gets its own MLP run.
        if M == 'M2' and dm.config.get('m2_transform') == 'local_zscore':
            mlp_key = (task_type, N, E, T, sample_idx, split)
        else:
            mlp_key = (task_type, N, T, sample_idx, split)
        if mlp_key not in mlp_cache:
            mlp_model = MLPModelFactory.create(
                task_type=task_type, in_dim=in_dim,
                hidden_dim=self.hidden_dim, out_dim=output_dimension,
                dropout=self.dropout,
            )
            mlp_trainer = GNNTrainer(
                model=mlp_model, device=self.device,
                lr=self.lr, weight_decay=self.weight_decay,
                epochs=self.epochs, patience=self.patience,
                task_type=task_type,
            )
            mlp_metrics_raw = mlp_trainer.train(
                data=data, num_classes=output_dimension,
                verbose=False, early_stopping=self.early_stopping,
            )
            s_mlp_raw, _ = compute_step1_score(mlp_metrics_raw, task_type)
            mlp_cache[mlp_key] = (s_mlp_raw, mlp_metrics_raw)
            self.logger.info(f"  MLP baseline trained: S_MLP={s_mlp_raw:.4f}" if not (isinstance(s_mlp_raw, float) and math.isnan(s_mlp_raw)) else "  MLP baseline trained: S_MLP=nan")

        s_mlp, mlp_metrics = mlp_cache[mlp_key]

        # M2 uses algebraic sigmoid to avoid (1-S_MLP) explosion and preserve
        # ordinal ranking for negative lift. M6 (derived) uses same formula.
        # All other tasks keep the original graph_lift formula.
        if M == 'M2':
            final_score = compute_final_score_algebraic(s_gnn, s_mlp)
        else:
            final_score = compute_final_score(s_gnn, s_mlp)

        main_row = build_result_row(
            variant=variant, test_metrics=gnn_metrics, data=data, df=df,
            trainer=gnn_trainer, run_split=split, sample_idx=sample_idx,
            output_dimension=output_dimension,
            s_gnn_step1=s_gnn, s_mlp_step1=s_mlp,
            final_score=final_score, r2_for_row=r2_for_row,
        )
        main_row['Predicted_Value'] = nan
        main_row['True_Value']      = nan

        # ── nan_reason tagging ───────────────────────────────────────────────
        if main_row.get('degenerate'):
            reasons = []
            if gnn_metrics.get('_raw_pred') is None:
                reasons.append('no_raw_scores')
            if math.isnan(s_gnn) and not math.isnan(r2_for_row) and r2_for_row == r2_for_row:
                pass  # s_gnn computed fine
            r2_val = gnn_metrics.get('r2', nan)
            if task_type in ('scalar', 'edge_scalar') and math.isnan(r2_val):
                reasons.append('ss_tot_guard')
            if task_type in ('global_categorical',) and math.isnan(s_gnn):
                reasons.append('tvd_baseline_guard')
            if task_type in ('global_scalar',) and math.isnan(s_gnn):
                reasons.append('gap_baseline_guard')
            if not reasons and gnn_trainer.is_degenerate(window=20, threshold=0.01):
                reasons.append('loss_plateau')
            if not reasons and math.isnan(final_score):
                reasons.append('no_raw_scores')
            main_row['nan_reason'] = ','.join(reasons)

        # ── M5 / M6 derived rows ─────────────────────────────────────────────
        derived_row = None
        if M in ('M1', 'M2'):
            gnn_gap = compute_aggregate_gap(gnn_metrics, task_type)
            mlp_gap = compute_aggregate_gap(mlp_metrics, task_type)

            if M == 'M1':
                s_gnn_d, _ = compute_m5_step1_score(gnn_gap, data)
                s_mlp_d, _ = compute_m5_step1_score(mlp_gap, data)
            else:
                s_gnn_d, _ = compute_m6_step1_score(gnn_gap, data)
                s_mlp_d, _ = compute_m6_step1_score(mlp_gap, data)

            # M6 (derived from M2) also uses algebraic lift
            if M == 'M2':
                final_score_d = compute_final_score_algebraic(s_gnn_d, s_mlp_d)
            else:
                final_score_d = compute_final_score(s_gnn_d, s_mlp_d)

            derived_row = build_derived_global_row(
                variant, gnn_gap, split, sample_idx, data, df, output_dimension,
                s_gnn_step1=s_gnn_d, s_mlp_step1=s_mlp_d, final_score=final_score_d,
            )
            if derived_row.get('degenerate'):
                if math.isnan(gnn_gap):
                    derived_row['nan_reason'] = 'no_raw_scores'
                elif M == 'M1' and math.isnan(s_gnn_d):
                    derived_row['nan_reason'] = 'tvd_baseline_guard'
                elif M == 'M2' and math.isnan(s_gnn_d):
                    derived_row['nan_reason'] = 'gap_baseline_guard'

        return main_row, derived_row


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run TAG experiments')
    parser.add_argument('--base_path',        default='data',   help='Path to data/ directory')
    parser.add_argument('--output_path',       default='output', help='Path to output/ directory')
    parser.add_argument('--hidden_dim',        type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs',            type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--lr',                type=float, default=DEFAULT_LR)
    parser.add_argument('--datasets',          nargs='+',  default=DATASETS)
    parser.add_argument('--device',            default=None)
    parser.add_argument('--verbose',           action='store_true')
    parser.add_argument('--early_stopping',    action='store_true')
    parser.add_argument('--n_train_samples',   type=int,   default=N_TRAIN_SAMPLES,
                        help='Number of train samples per variant (default 10)')
    parser.add_argument('--n_test_samples',    type=int,   default=N_TEST_SAMPLES,
                        help='Number of test samples per variant (default 10)')
    parser.add_argument('--record_wall_time',  action='store_true',
                        help='Optional: add a wall_time_seconds column, one value per '
                             'variant (total time across all its train+test samples), '
                             'to construction_performance_table. OFF by default -- when '
                             'off, output is identical to before this flag existed.')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Optional: restrict to these M-task types only (e.g. --tasks M1). '
                             'Unset = every task type, unchanged default.')
    parser.add_argument('--epoch_log_variants', default=None,
                        help='Optional: comma-separated list of FULL variant identifiers '
                             'including the task prefix ("M1/N7/E10b/T12a" -- matches '
                             'variant_str = f"{M}/{N}/{E}/{T}" exactly; a 3-part '
                             '"N7/E10b/T12a" will silently never match) to enable '
                             'per-epoch logging for. Logs go to '
                             '{output_path}/epoch_logs/{variant}_{sample}.csv. Variants '
                             'not listed are unaffected -- no epoch logging by default.')
    args = parser.parse_args()

    epoch_log_variants = (args.epoch_log_variants.split(',') if args.epoch_log_variants
                          else None)

    runner = ExperimentRunner(
        base_path=args.base_path,
        output_path=args.output_path,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
        datasets=args.datasets,
        n_train_samples=args.n_train_samples,
        n_test_samples=args.n_test_samples,
        record_wall_time=args.record_wall_time,
        epoch_log_variants=epoch_log_variants,
        tasks=args.tasks,
    )
    runner.run()


if __name__ == '__main__':
    # When running from the repo root, add code/ to sys.path
    sys.path.insert(0, str(Path(__file__).parent))
    main()
