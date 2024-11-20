#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72 
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --mem=40G
#SBATCH -p fat_rome
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

echo "job started"

source requirements/load_venv.sh
srun python -m pop2vec.llm.src.new_code.pipeline pop2vec/llm/projects/dutch_real/pipeline_no_mlm_snellius_cfg.json





