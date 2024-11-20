#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72 
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --mem=20G
#SBATCH -p fat_rome
#SBATCH -e /home/tislampial/logs/%x-%j.err
#SBATCH -o /home/tislampial/logs/%x-%j.out	

echo "job started"

date
source requirements/load_venv.sh
srun python -m pop2vec.llm.src.new_code.pipeline pop2vec/llm/projects/dutch_real/pipeline_oct_24.json
echo "job ended"

