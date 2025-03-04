#!/bin/bash
#SBATCH --job-name=run_clustering
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH -p comp_env
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

echo "Job started"

source requirements/load_venv.sh

python -m pop2vec.clustering.run_clustering

echo "job ended."
