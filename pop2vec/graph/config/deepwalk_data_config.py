from pop2vec.utils.constants import DATA_ROOT

data_config = {
    "parquet_root": DATA_ROOT + "/graph/walks/",
    "parquet_nests": "*/*/*/*/*.parquet",
    "walk_iteration_name": "walklen40_prob0.8",
    "embedding_dir": DATA_ROOT + "/graph/embeddings/",
    "model_dir": DATA_ROOT + "/graph/models/",
    "mapping_dir": DATA_ROOT + "/graph/mappings/",
}
