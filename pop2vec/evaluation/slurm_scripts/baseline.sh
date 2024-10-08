#!/bin/bash
#
#SBATCH --job-name=base_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=15G
#SBATCH --array=1-8
#SBATCH -e logs/%x.%A_%a.err
#SBATCH -o logs/%x.%A_%a.out

echo "Job started"

# Assumes you're in the project repository
CONFIG_FILE="pop2vec/evaluation/slurm_scripts/base_eval_config.txt"

# Read the line corresponding to the current array task ID
LINE=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==ArrayTaskID {print; exit}' "$CONFIG_FILE")

# Check if LINE is empty
if [ -z "$LINE" ]; then
    echo "No configuration found for ArrayTaskID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse the parameters
IFS=$'\t' read -r ArrayTaskID SampleName TrainOnly OutputDir EmbeddingsPath EmbeddingType <<< "$LINE"

echo "Running job with the following parameters:"
echo "ArrayTaskID: $ArrayTaskID"
echo "SampleName: $SampleName"
echo "TrainOnly: $TrainOnly"
echo "OutputDir: $OutputDir"
echo "EmbeddingsPath: $EmbeddingsPath"
echo "EmbeddingType: $EmbeddingType"

date
source requirements/load_venv.sh

# Build the command
CMD="python -m pop2vec.evaluation.baseline_evaluation_v1 --predictor-year 2016"

# Add train-only flag if necessary
if [ "$TrainOnly" -eq "1" ]; then
    CMD+=" --train-only"
fi

# Add output directory
CMD+=" --output-dir $OutputDir"

# Add embeddings arguments if embeddings are used
if [ "$EmbeddingsPath" != "None" ] && [ "$EmbeddingType" != "None" ]; then
    CMD+=" --embeddings-path $EmbeddingsPath --embedding-type $EmbeddingType"
fi

echo "Executing command:"
echo "$CMD"

# Execute the command
time $CMD

echo "Job ended"
