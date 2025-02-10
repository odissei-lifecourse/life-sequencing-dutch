#!/bin/bash
#
#SBATCH --job-name=Edge_16
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH -p comp_env
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

source requirements/load_venv.sh

export CUDA_VISIBLE_DEVICES=3

ndim=128
window_size=10
max_epochs=50

# NOTE: `--gpus` should be 0 as long as using a single device. The argument
# value is not tied to the index of the visible devices set above.

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



echo "job ended"
