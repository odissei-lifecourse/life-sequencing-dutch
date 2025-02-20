#!/bin/bash
#
#SBATCH --job-name=test_infer
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --mem=10G
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

#declare PREFIX="/gpfs/ostor/ossc9424/homedir/"

#export CUDA_VISIBLE_DEVICES=0

echo "job started"

source requirements/load_venv.sh

date
time python -m pop2vec.llm.src.new_code.infer_embedding pop2vec/llm/configs/Snellius/infer_cfg_test_snellius.json

echo "job ended successfully"
