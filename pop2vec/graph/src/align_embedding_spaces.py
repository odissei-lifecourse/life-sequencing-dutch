"""Embedding alignment utilities for temporal DeepWalk snapshots.

The script reads a configuration JSON that specifies a target embedding file
and a list of source embedding files.  Each embedding file is a Parquet file
with the following schema:

    - rinpersoon_id:  Unique identifier (int64 or string) for a node/person.
    - emb_0 .. emb_{K-1}:  Float32/Float64 embedding dimensions.

It performs one or more alignment strategies (orthogonal Procrustes, scaled
Procrustes, or ordinary least‑squares with intercept) so that every source
embedding lives in the same space as the *target* embedding.  The aligned
embeddings are written back to *target_dir* using the naming convention

    <original_path_with_slashes_replaced_by_dashes>_<strategy>.parquet

It also evaluates the quality of each alignment on a 1% validation set by
computing Pearson and Spearman correlations between cosine similarities in
the target space and in the aligned source space.

Everything is organised in small, testable classes that follow Google's
Python style guide.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.stats import pearsonr, spearmanr

import csv

# Static random seed for reproducibility unless overridden via CLI.
DEFAULT_SEED: int = 42

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns cosine similarity between two 1‑D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def format_output_path(original_path: str, target_dir: pathlib.Path, suffix: str
                       ) -> pathlib.Path:
    """Creates the output path inside *target_dir* following the specification."""
    name = original_path.lstrip(os.sep).replace(os.sep, "-")
    return target_dir / f"{name}_{suffix}.parquet"


# -----------------------------------------------------------------------------
# Embedding loaders & samplers
# -----------------------------------------------------------------------------

class EmbeddingTable:
    """In‑memory representation of an embedding snapshot."""

    def __init__(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.ids = ids              # shape: (N,)
        self.vectors = vectors      # shape: (N, K)
        self._index: dict[int, int] | None = None

    @classmethod
    def from_parquet(
        cls, 
        path: str | pathlib.Path, 
        fraction: float = 1.0, 
        seed: int = DEFAULT_SEED, 
        id_col: str = "rinpersoon_id",
    ) -> "EmbeddingTable":
        """Loads a fraction of the parquet file into memory.

        Args:
          path: Parquet file location.
          fraction: Fraction of rows to load (0 < fraction ≤ 1).
          seed: PRNG seed for sampling.
        """
        logging.info("Reading parquet: %s", path)
        df = pd.read_parquet(path, columns=None)  # Load all columns; filter later.
        logging.info("Loaded %d rows.", len(df))

        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1].")

        if fraction < 1:
            df = df.sample(frac=fraction, random_state=seed)
            logging.info("Sub-sampled to %d rows (fraction=%.3f).", len(df), fraction)

        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        ids = df[id_col].to_numpy()
        vectors = df[emb_cols].to_numpy(dtype=np.float32)
        return cls(ids, vectors)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def restrict_to_ids(self, ids: Sequence[int | str]) -> "EmbeddingTable":
        """Returns a *view* containing only the specified IDs, order preserved."""
        if self._index is None:
            self._index = {id_: i for i, id_ in enumerate(self.ids)}
        idx = [self._index[id_] for id_ in ids]
        return EmbeddingTable(self.ids[idx], self.vectors[idx])


# -----------------------------------------------------------------------------
# Alignment strategy abstractions
# -----------------------------------------------------------------------------

class AlignmentModel:
    """Base class for every alignment strategy."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class OrthogonalProcrustes(AlignmentModel):
    """Pure rotation / reflection alignment (no scaling)."""

    def __init__(self) -> None:
        self._w: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Finds the orthogonal matrix W that minimises ||XW-Y||_F."""
        if x.shape != y.shape:
            raise ValueError("Shapes of x and y must match.")
        logging.debug("Computing cross‑covariance for Procrustes.")
        c = y.T @ x  # (K, K)
        u, _, vt = svd(c, full_matrices=False)
        self._w = u @ vt
        logging.debug("Orthogonal matrix W computed.")

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("Model is not fitted.")
        return x @ self._w


class ScaledOrthogonalProcrustes(AlignmentModel):
    """Rotation + uniform scaling alignment (similarity Procrustes)."""

    def __init__(self) -> None:
        self._w: np.ndarray | None = None
        self._s: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape != y.shape:
            raise ValueError("Shapes of x and y must match.")
        c = y.T @ x
        u, sigma, vt = svd(c, full_matrices=False)
        w = u @ vt
        s = sigma.sum() / np.square(x).sum()
        self._w = w
        self._s = float(s)
        logging.debug("Scaled Procrustes parameters computed (s=%.4f).", self._s)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._w is None or self._s is None:
            raise RuntimeError("Model is not fitted.")
        return self._s * (x @ self._w)


class OrdinaryLeastSquares(AlignmentModel):
    """KxK linear map + intercept (bias) for each dimension."""

    def __init__(self) -> None:
        self._w: np.ndarray | None = None  # shape: (K+1, K) including bias row.

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        logging.debug("Fitting OLS with intercept.")
        n, k = x.shape
        ones = np.ones((n, 1), dtype=x.dtype)
        design = np.hstack((x, ones))  # (n, K+1)
        # Solve design @ W = y  =>  W = (design^T design)^‑1 design^T y
        w, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._w = w  # (K+1, K)
        logging.debug("OLS coefficients fitted.")

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("Model is not fitted.")
        n = x.shape[0]
        ones = np.ones((n, 1), dtype=x.dtype)
        design = np.hstack((x, ones))
        return design @ self._w


# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

@dataclass
class CorrelationMetrics:
    pearson_corr: float
    pearson_pval: float
    spearman_corr: float
    spearman_pval: float


class Evaluator:
    """Computes correlation metrics on a validation set."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def _pairwise_cosines(
        self, vectors: np.ndarray, permuted_vectors: np.ndarray
    ) -> np.ndarray:
        """Vectorised cosine similarity for two aligned matrices."""
        dot = np.sum(vectors * permuted_vectors, axis=1)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(permuted_vectors, axis=1)
        return dot / norms

    def correlations(
        self,
        target_vectors: np.ndarray,
        aligned_vectors: np.ndarray,
    ) -> CorrelationMetrics:
        n = target_vectors.shape[0]
        permutation = self._rng.permutation(n)
        # Pair i‑th row with permutation[i].
        cos_target = self._pairwise_cosines(target_vectors, target_vectors[permutation])
        cos_aligned = self._pairwise_cosines(aligned_vectors, aligned_vectors[permutation])
        pearson = pearsonr(cos_target, cos_aligned, alternative='greater')
        pearson_corr, pearson_pval = pearson.statistic, pearson.pvalue
        spearman = spearmanr(cos_target, cos_aligned, alternative='greater')
        spearman_corr, spearman_pval = spearman.statistic, spearman.pvalue
        return CorrelationMetrics(pearson_corr, pearson_pval, spearman_corr, spearman_pval)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

STRATEGY_MAP = {
    "op": OrthogonalProcrustes,
    "op_scale": ScaledOrthogonalProcrustes,
    "ols": OrdinaryLeastSquares,
}


def run_alignment(
    target_path: str,
    source_paths: Sequence[str],
    strategy_names: Sequence[str],
    target_dir: str | pathlib.Path,
    fraction: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    # 1. Load *target* snapshot and subsample ``fraction`` rows.
    logging.info("Loading target snapshot and sampling fraction %.3f", fraction)
    target_table_full = EmbeddingTable.from_parquet(target_path, fraction=1.0, seed=seed)
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0,1].")
    sample_size = max(1, math.ceil(fraction * len(target_table_full.ids)))
    sample_idx = rng.choice(len(target_table_full.ids), size=sample_size, replace=False)
    target_sample = EmbeddingTable(target_table_full.ids[sample_idx],
                                   target_table_full.vectors[sample_idx])

    k = target_sample.dim
    logging.info("Target embedding dimensionality: %d", k)

    # Build global validation set: intersection of IDs across all files.
    logging.info("Computing 1%% validation set intersection across files.")
    target_ids_set = set(target_sample.ids)
    for path in source_paths:
        ids = pd.read_parquet(path, columns=["rinpersoon_id"])["rinpersoon_id"].to_numpy()
        target_ids_set &= set(ids)
    ids_intersection = np.fromiter(target_ids_set, dtype=target_sample.ids.dtype)
    rng.shuffle(ids_intersection)
    val_size = max(1, math.ceil(0.01 * len(ids_intersection)))
    val_ids = ids_intersection[:val_size]
    train_ids = ids_intersection[val_size:]
    logging.info("Validation set size: %d; Training IDs available in all files: %d",
                 len(val_ids), len(train_ids))

    # Prepare evaluator with fixed RNG.
    evaluator = Evaluator(rng)

    # Ensure target_dir exists.
    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []


    tgt_train = target_sample.restrict_to_ids(train_ids)
    # Validation vectors from target (same for every source).
    target_val_vectors = (
        target_sample.restrict_to_ids(val_ids).vectors.astype(np.float32)
    ) 
    fieldnames = [
        "source",
        "target",
        "validation_size",
        "strategy",
        "pearson",
        "spearman",
        "pearson_pval",
        "spearman_pval"
    ]
    with open(target_dir / "alignment_metrics.csv", mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for src_path in source_paths:
            logging.info("Processing source %s", src_path)
            # Load source table (fraction of rows).
            src_table = EmbeddingTable.from_parquet(src_path, fraction=1.0, seed=seed)
            # Align training IDs intersection only.
            src_train = src_table.restrict_to_ids(train_ids)
            

            for strategy_name in strategy_names:
                if strategy_name not in STRATEGY_MAP:
                    raise ValueError(f"Unknown strategy '{strategy_name}'.")
          
                StrategyCls = STRATEGY_MAP[strategy_name]

            
                model = StrategyCls()
                model.fit(src_train.vectors, tgt_train.vectors)

                # Align *all* rows of the *full* source (so we can save the file).
                logging.info("aligning vectors")
                aligned_vectors_full = model.transform(src_table.vectors)

                # Save aligned snapshot.
                out_path = format_output_path(src_path, target_dir, strategy_name)
                logging.info("Saving aligned file to %s", out_path)
                aligned_df = pd.DataFrame(aligned_vectors_full,
                                          columns=[f"emb_{i}" for i in range(k)])
                aligned_df.insert(0, "rinpersoon_id", src_table.ids)
                aligned_df.to_parquet(out_path, engine="pyarrow", index=False)
                logging.info("File saved!")
                # Evaluate on validation set.
                src_val_vectors = model.transform(
                    src_table.restrict_to_ids(val_ids).vectors.astype(np.float32)
                )
                metrics = evaluator.correlations(target_val_vectors, src_val_vectors)
                # logging.info("Source %s -> Pearson %.4f (p-val %.4f), Spearman %.4f (p-val %.4f)",
                #              src_path, metrics.pearson_corr, metrics.spearman_corr, )
                row = {
                    "source": src_path,
                    "target": target_path,
                    "validation_size": len(val_ids),
                    "strategy": strategy_name,
                    "pearson": metrics.pearson_corr,
                    "spearman": metrics.spearman_corr,
                    "pearson_pval": metrics.pearson_pval,
                    "spearman_pval": metrics.spearman_pval
                }
                logging.info(row)
                writer.writerow(row)


    # # Write metrics CSV.
    # metrics_df = pd.DataFrame(metrics_rows)
    # metrics_path = target_dir / "alignment_metrics.csv"
    # metrics_df.to_csv(metrics_path, index=False)
    # logging.info("Metrics written to %s", metrics_path)


# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align DeepWalk snapshots.")
    parser.add_argument("--config", type=str,
                        help="Path to JSON configuration file.")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of *source* embeddings to load (default 1.0).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default 42).")
    parser.add_argument("--loglevel", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default INFO).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.loglevel,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y‑%m‑%d %H:%M:%S")

    # Read configuration JSON.
    with open(args.config, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    required_keys = {"target_path", "source_paths", "alignment_strategy", "target_dir"}
    if not required_keys.issubset(cfg):
        missing = required_keys - set(cfg)
        raise KeyError(f"Missing keys in config: {missing}.")

    run_alignment(
        target_path=cfg["target_path"],
        source_paths=cfg["source_paths"],
        strategy_names=cfg["alignment_strategy"],
        target_dir=cfg["target_dir"],
        fraction=args.fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
