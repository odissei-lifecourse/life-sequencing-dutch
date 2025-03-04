import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from pop2vec.clustering.core import estimate_and_evaluate
from pop2vec.utils.sample_from_parquet import sample_ids_from_parquet

data_dir = "/gpfs/ostor/ossc9424/homedir/data/graph/embeddings/"
emb_path = "year=2016/iter_name=walklen40_prob0.8/record_edge_type=1/embedding_50.parquet"

SAMPLE_SIZE = 50_000

N_CLUSTERS = [10, 25, 50, 100, 150]

if __name__ == "__main__":
    embs = sample_ids_from_parquet(data_dir + emb_path, n=SAMPLE_SIZE)

    input_map = {
        "kmeans": KMeans(random_state=0, n_init="auto"),
        "agglom_base": AgglomerativeClustering(),
        "agglom_cosine": AgglomerativeClustering(metric="cosine", linkage="average"),
        "gmm": GaussianMixture(),
    }

    all_results = []
    for n_clusters in tqdm(N_CLUSTERS):
        current_results = {}

        for name, estimator in input_map.items():
            current_results[name] = estimate_and_evaluate(name, estimator, embs, n_clusters)

        score_dfs = []
        for k, v in current_results.items():
            df_temp = pl.DataFrame(v["scores"]).unpivot(variable_name="metric", value_name="value")
            df_temp = df_temp.with_columns(pl.lit(k).alias("method"))
            score_dfs.append(df_temp)

        score_dfs = pl.concat(score_dfs)
        score_dfs = score_dfs.with_columns(pl.lit(n_clusters).alias("n_clusters"))

        all_results.append(score_dfs)

    all_results = pl.concat(all_results)

    all_results.write_csv("clustering_performance.csv")
