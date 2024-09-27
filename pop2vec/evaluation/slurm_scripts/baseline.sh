#!/bin/bash
#
#SBATCH --job-name=base_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=15G
#SBATCH --array=1-2
#SBATCH -e logs/%x.%A_%a.err
#SBATCH -o logs/%x.%A_%a.out

echo "job started"

data_dir="/projects/0/prjs1019/data/"

# assumes you're in the project repository
config="pop2vec/evaluation/slurm_scripts/base_eval_config.txt"

train_only=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $3}' $config)


date
source requirements/load_venv.sh

if [ "$train_only" -eq "1" ]; then
    time python -m pop2vec.evaluation.baseline_evaluation_v1 --data-dir $data_dir --train-only
elif [ "$train_only" -eq "0" ]; then
    time python -m pop2vec.evaluation.baseline_evaluation_v1 --data-dir $data_dir
fi

    
echo "job ended"
