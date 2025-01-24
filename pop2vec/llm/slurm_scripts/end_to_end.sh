#!/bin/bash
#
#SBATCH --job-name=end2end
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 64 
#SBATCH --time=03:00:00
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out
#SBATCH --partition=fat_rome
#SBATCH --mem=400G

echo "job started"

date
pwd
source requirements/load_venv.sh

echo "creating fake data"
#time python -m pop2vec.fake_data.generate_step2_data

echo "step 3: categorical transformation"
time python -m pop2vec.llm.src.new_code.preprocess_data pop2vec/llm/projects/dutch_real/preprocess_cfg.json

echo "step 4: life sequences"
time python -m pop2vec.llm.src.new_code.create_life_seq_parquets pop2vec/llm/projects/dutch_real/create_parquet_seq_cfg.json

echo "step 5: training data"
srun python -m pop2vec.llm.src.new_code.pipeline pop2vec/llm/projects/dutch_real/pipeline_cfg.json
srun python -m pop2vec.llm.src.new_code.pipeline pop2vec/llm/projects/dutch_real/pipeline_no_mlm_cfg.json


echo "job ended successfully"
