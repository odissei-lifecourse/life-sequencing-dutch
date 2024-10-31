#!/bin/bash
#
#SBATCH --job-name deepwalk
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=45:00:00
#SBATCH -p comp_env
#SBATCH --mem=50GB
#SBATCH --nodelist=ossc9424vm4
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

ndim=128
window_size=10
max_epochs=50

source requirements/load_venv.sh

python -m pop2vec.graph.src.deepwalk \
    --dim "$ndim" \
    --window_size "$window_size" \
    --num_walks 1 \
    --only_gpu \
    --gpus 0 \
    --print_loss \
    --start_index 0 \
    --max_epochs "$max_epochs" \
    --year 2016 \
    --record_edge_type
