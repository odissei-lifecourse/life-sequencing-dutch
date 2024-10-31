
Documentation for graph data.

### Generating random walks

The random walks that record the edge types are created with the code in the separate [repository](https://github.com/odissei-lifecourse/layered_walk).
The code there creates a parquet dataset with the following file partitioning
- `year=YYYY`
- `iter_name=some_name`
- `record_edge_type=X`
- `dry=0`
- `/chunk-i.parquet`. In each iteration `i`, one walk is created for all nodes and all layers in the network. `X` is a boolean whether the walks have the edge types recorded or not.

To generate the walks, the user needs to give a name to this particular iteration, which is recorded in the `iter_name` features. If iterations are repeated, existing walks are overwritten.

If you create a new partitioning, make sure to use the same convention `partition=X`, and update the `ParquetWalk` dataclass described below.

### Training `deepwalk`

The `deepwalk_dataset` uses the `ParquetWalk` class from `pop2vec.utils.parquet_walks`. This class handles the partitioning described
above and loads the walks for one epoch into a Dataframe.
The code for `deepwalk` requires some command-line arguments as per the script `deepwalk.py`.
In addition, it requires the file `pop2vec.graph.config.data_config`, where the `deepwalk_data_config` dict defines data paths:
- Where to store the model and the embeddings
- The nesting structure of the parquet file described above -- this is important because the `ParquetWalk` class needs this information to query the entire set of walks.
- The name of the iteration that generated the walks

When `deepwalk` is run, models and embeddings are stored in the same structure as the walks, starting from their respective roots. This means that
- when deepwalk is run with repeated configurations, models and embeddings are overwritten
- even if deepwalk is run with different hyperparameters, because not all hyperparameters are stored in the file name, models and embeddings may be overwritten.

An example of a slurm script to run deepwalk is in `pop2vec.graph.slurm_scripts.run_deepwalk.py`.
