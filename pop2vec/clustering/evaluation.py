from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def summarise_cluster(cluster_df):
    """Summary statistics for clusters created."""
    return (
        cluster_df.group_by("cluster")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("id").min().alias("min_id"),
                pl.col("id").max().alias("max_id"),
                pl.col("id").mean().alias("avg_id"),
            ]
        )
        .sort("cluster")
    )


def plot2d(cluster_df: pl.DataFrame, embs_df: pl.DataFrame) -> Figure:
    """Plot clusters in 2d.

    Reduce embeddings to 2 dimensions with TSNE, and plot in this space,
    with coloring according to clusters.

    Args:
        cluster_df (pl.DataFrame): dataframe with person IDs and assigned clusters.
        embs_df (pl.DataFrame): dataframe with person IDs and embeddings.

    Returns:
        plt.Figure

    """
    x = embs_df[:, 1:]
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
    x_proj = tsne.fit_transform(x)
    plot_df = pl.DataFrame(x_proj, schema=["x_coord", "y_coord"])
    plot_df = plot_df.with_columns(embs_df.select("rinpersoon_id"))
    plot_df = plot_df.with_columns(cluster_df.select("cluster"))

    pdf = plot_df.to_pandas()

    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pdf["x_coord"], pdf["y_coord"], c=pdf["cluster"], cmap="viridis")

    plt.colorbar(scatter, label="Cluster ID")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("2D Scatter Plot of Clusters")
    plt.tight_layout()

    return fig  # type: ignore[attr-defined]


def barplot_cluster_sizes(cluster_df: pl.DataFrame) -> Figure:
    """Create a bar plot of cluster sizes and return the figure object for further handling.

    Parameters:
    -----------
    cluster_df : DataFrame
        DataFrame containing cluster data

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object which can be displayed or saved
    """
    summary = summarise_cluster(cluster_df)

    cluster_ids = summary["cluster"].to_numpy()
    counts = summary["count"].to_numpy()

    # Create figure explicitly and get the figure object
    fig = plt.figure(figsize=(8, 4))

    # Create the plot on the current figure
    plt.bar(cluster_ids, counts, color="skyblue", edgecolor="navy")

    plt.title("Number of Items per Cluster", fontsize=15)
    plt.ylabel("Count", fontsize=12)
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Apply tight layout to ensure proper spacing
    plt.tight_layout()

    # Return the figure object
    return fig  # type: ignore[attr-defined]


def fraction_closest_own_centroid(units_emb, cluster_emb, cluster_df):
    """Compute fraction of units closest to own centroid.

    For unit-level and cluster-level embeddings,
    compute fraction of units whose closest cluster centroid is the centroid
    of their own cluster.

    """
    cos_sim = cosine_similarity(units_emb, cluster_emb)

    argmaxes = np.argmax(cos_sim, axis=1)
    closest_to_own_centroid = np.nonzero(argmaxes == cluster_df["cluster"].to_numpy())
    return closest_to_own_centroid[0].shape[0] / argmaxes.shape[0]


def score_assigned_clusters(x, labels):
    """Evaluate the assigned clusters.

    Args:
        x: array of embeddings
        labels: array of cluster IDs
        https://scikit-learn.org/stable/modules/clustering.html.
    """
    score_map = {
        "silhouette": lambda x, y: silhouette_score(x, y, metric="cosine"),
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
    }
    return {k: f(x, labels) for k, f in score_map.items()}


def pairwise_comparison(x, clusters, sample_size=None):
    """Distribution of cosine similarities within and across clusters.

    Compare each pair within X, own-pair exclusive.

    X: array of embeddings
    clusters: array of cluster ids
    assumed in same order
    """
    rng = np.random.default_rng()
    n = x.shape[0]
    if sample_size and sample_size < n:
        samples = rng.choice(np.arange(n), size=sample_size, replace=False)
        x = x[samples]
        clusters = clusters[samples]

    cos_sim = cosine_similarity(x)

    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    one_hot = np.eye(len(unique_clusters), dtype=bool)[inverse]
    group_matrix = np.dot(one_hot, one_hot.T).astype(int)
    triu1 = np.triu_indices_from(group_matrix, k=1)

    same_cluster = group_matrix[triu1]
    similarity = cos_sim[triu1]

    out = np.vstack([same_cluster, similarity]).T
    return pl.DataFrame(out, schema=["same_cluster", "similarity"])


def make_pairwise_histogram(df_compare):
    """Plot pairwise distances for points in- and between clusters."""
    fig = plt.figure(figsize=(8, 4))
    plt.hist(
        [
            df_compare.filter(pl.col("same_cluster") == 0).select("similarity").to_numpy().squeeze(),
            df_compare.filter(pl.col("same_cluster") == 1).select("similarity").to_numpy().squeeze(),
        ],
        label=["Different cluster", "Same cluster"],
        bins=40,
        density=True,
    )
    plt.xlabel("Similarity")
    plt.ylabel("Density")
    plt.title("Histogram of Similarities by Cluster membership")
    plt.legend()
    plt.tight_layout()
    return fig
