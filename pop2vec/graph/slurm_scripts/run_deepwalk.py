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


source requirements/load_venv.sh

python -m pop2vec.graph.src.deepwalk \
        --dim 128 \
        --window_size 10 \
        --num_walks 1 \
        --only_gpu \
        --gpus 0 \
        --print_loss \
        --start_index 0 \
        --max_epochs 50 \
        --year 2016
