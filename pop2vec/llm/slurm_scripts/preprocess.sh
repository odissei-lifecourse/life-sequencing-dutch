#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/preprocess_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/preprocess_stdout.txt

echo "job started"

date
python pop2vec/llm/src/new_code/preprocess_data.py ppop2vec/llm/rojects/dutch_real/preprocess_cfg.json

echo "job ended successfully"