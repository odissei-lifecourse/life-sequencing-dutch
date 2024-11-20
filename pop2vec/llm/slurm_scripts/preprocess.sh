#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH -e /home/tislampial/logs/%x.%j.err
#SBATCH -o /home/tislampial/logs/%x.%j.out
#SBATCH --partition=rome
#SBATCH --mem=30G

echo "job started"

date
pwd
source requirements/load_venv.sh
python -m pop2vec.llm.src.new_code.preprocess_data pop2vec/llm/projects/dutch_real/preprocess_cfg.json

echo "job ended successfully"
