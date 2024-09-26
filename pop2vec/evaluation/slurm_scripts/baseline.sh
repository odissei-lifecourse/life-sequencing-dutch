#!/bin/bash
#
#SBATCH --job-name=base_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=15G
#SBATCH -e logs/%x.%j.err
#SBATCH -o logs/%x.%j.out

echo "job started"

data_dir="/projects/0/prjs1019/data/"
date
source requirements/load_venv.sh
time python -m pop2vec.evaluation.baseline_evaluation_v1 $data_dir

echo "job ended"
