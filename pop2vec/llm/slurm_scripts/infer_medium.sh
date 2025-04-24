#!/bin/bash
#
#SBATCH --job-name=infer_medium
#SBATCH --ntasks-per-node=3
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=130G
#SBATCH -p comp_env
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out


export CUDA_VISIBLE_DEVICES=0

echo "job started"

source requirements/load_venv.sh


date
python -m pop2vec.llm.src.new_code.infer_embedding pop2vec/llm/configs/OSSC/infer_cfg_medium.json

echo "job ended successfully"
