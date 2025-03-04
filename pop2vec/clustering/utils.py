from __future__ import annotations
import numpy as np
import polars as pl


def create_fake_embs(n=1_000, k=128, n_clusters=5):
    """Create fake embedding vectors."""
    rng = np.random.default_rng()
    cluster_centers = rng.normal(size=(n_clusters, k))
    cluster_assignments = rng.integers(0, n_clusters, size=n)

    selected_centers = cluster_centers[cluster_assignments]  # Shape: (n, k)
    noise = rng.normal(scale=1, size=(n, k))
    embs = selected_centers + noise

    # Generate random IDs
    ids = rng.choice(np.arange(1_000_000, 100_000_000), n, replace=False)
    ids.dtype = np.int64

    # Combine IDs with embeddings
    data = np.hstack([np.expand_dims(ids, 1), embs])
    names = ["rinpersoon_id"] + [f"emb_{i}" for i in range(k)]

    return pl.DataFrame(data, schema=names)
