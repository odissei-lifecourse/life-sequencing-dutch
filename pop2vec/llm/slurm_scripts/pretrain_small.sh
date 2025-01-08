#!/bin/bash
#
#SBATCH --job-name=pretrain_small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

source requirements/load_venv.sh

#export CUDA_VISIBLE_DEVICES=0

date
time python -m pop2vec.llm.src.new_code.pretrain --config=pop2vec/llm/projects/dutch_real/pretrain_cfg_small.json

echo "job ended successfully"
