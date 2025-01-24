#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64 
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=400G
#SBATCH -p fat_rome
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

echo "job started"

source requirements/load_venv.sh
#srun python -m pop2vec.llm.src.new_code.pipeline # pop2vec/llm/projects/dutch_real/pipeline_cfg.json
srun python -m pop2vec.llm.src.new_code.pipeline DO_MLM=false # pop2vec/llm/projects/dutch_real/pipeline_no_mlm_cfg.json





