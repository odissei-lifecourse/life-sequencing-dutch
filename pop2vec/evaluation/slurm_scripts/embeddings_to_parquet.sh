#!/bin/bash
#
#SBATCH --job-name=emb_to_pq
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=250G
#SBATCH -p comp_env
#SBATCH -e logs/$x-%j.err
#SBATCH -o logs/%x-%j.out

source requirements/load_venv.sh
python -m pop2vec.utils.convert_hdf5_to_parquet
