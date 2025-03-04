from __future__ import annotations
from typing import Union
import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from pop2vec.clustering.evaluation import barplot_cluster_sizes
from pop2vec.clustering.evaluation import fraction_closest_own_centroid
from pop2vec.clustering.evaluation import make_pairwise_histogram
from pop2vec.clustering.evaluation import pairwise_comparison
from pop2vec.clustering.evaluation import plot2d
from pop2vec.clustering.evaluation import score_assigned_clusters

ClusterEstimator = Union[AgglomerativeClustering, KMeans, GaussianMixture]


def get_clusters(estimator, n_clusters, data):
    """Compute clusters.

    Args:
        estimator: An instance of clustering estimator from sklearn.cluster
        n_clusters: Number of clusters to find.
        data: polars dataframe with ids in column 0 and features in remaining columns.
    """
    ids, x = (data[:, 0], data[:, 1:])
    if isinstance(estimator, GaussianMixture):
        estimator.n_components = n_clusters
    else:
        estimator.n_clusters = n_clusters

    estimator.fit(x)
    if isinstance(estimator, GaussianMixture):
        probs = estimator.predict_proba(x)
        cluster_ids = np.argmax(probs, axis=1)
        clusters = np.vstack([ids, cluster_ids]).T
    else:
        clusters = np.vstack([ids, estimator.labels_]).T

    cluster_df = pl.DataFrame(clusters, schema=["id", "cluster"], orient="row")

    # Cast columns to appropriate types
    cluster_df = cluster_df.with_columns([pl.col("id").cast(pl.Int64), pl.col("cluster").cast(pl.Int64)])

    centroid_schema = ["cluster", *x.columns]
    if isinstance(estimator, (AgglomerativeClustering, GaussianMixture)):
        clf = NearestCentroid()
        labels = clusters[:, 1]
        clf.fit(x, labels)
        centroids = clf.centroids_
    elif isinstance(estimator, KMeans):
        centroids = estimator.cluster_centers_
    else:
        raise NotImplementedError

    centroids = [np.expand_dims(np.arange(n_clusters), 1), centroids]
    centroids = pl.DataFrame(np.hstack(centroids), schema=centroid_schema)
    return cluster_df, centroids


def estimate_and_evaluate(name: str, estimator: ClusterEstimator, embs: pl.DataFrame, n_clusters):
    """Estimate clusters and evaluate."""
    cluster_df, centroids = get_clusters(estimator, n_clusters, embs)
    plt = barplot_cluster_sizes(cluster_df)
    plt.savefig(f"sizes_{name}.png")
    coverage = fraction_closest_own_centroid(embs[:, 1:].to_numpy(), centroids[:, 1:].to_numpy(), cluster_df)

    plt = plot2d(cluster_df, embs)
    plt.savefig(f"plot2d_{name}.png")

    ## evaluation
    x = embs[:, 1:].to_numpy()
    labels = cluster_df["cluster"].to_numpy()

    scores = score_assigned_clusters(x, labels)

    df_compare = pairwise_comparison(x, labels, sample_size=2_000)
    plt = make_pairwise_histogram(df_compare)
    plt.savefig(f"dist-hist_{name}.png")

    return {"coverage": coverage, "cluster_df": cluster_df, "centroids": centroids, "scores": scores}
