#!/bin/bash
#
#SBATCH --job-name=isolate_evaluation_subset
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=200GB
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err

cd /gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/

module purge 
source ossc_env/bin/activate
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-intel-2022a

cd /gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/

date
echo "Starting scripts"
time python isolate_income_subset.py
time python isolate_marriage_subset.py 

# subset embeddings for different models
echo "Extracting subset for llm new"
time python extract_embedding_subset.py --model llm_new

echo "Extracting subset for llm old"
time python convert_embeddings_to_hdf5.py 
time python extract_embedding_subset.py --model llm_old 

echo "Extracting subset for network"
time python convert_pickle_embeddings.py 
time python extract_embedding_subset.py --model network


echo "job ended" 