#!/bin/bash
#
#SBATCH --job-name=create_parquet_seq
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH -p rome 
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out


echo "job started"


date
source requirements/load_venv.sh
time python -m pop2vec.llm.src.new_code.create_life_seq_parquets pop2vec/llm/projects/dutch_real/create_parquet_seq_cfg.json

echo "job ended"


