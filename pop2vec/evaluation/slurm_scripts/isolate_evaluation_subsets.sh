#!/bin/bash
#
#SBATCH --job-name=isolate_evaluation_subset
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=200GB
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err

cd /gpfs/ostor/ossc9424/homedir/

module purge 
source ossc_env/bin/activate
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-intel-2022a

cd /gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/

date
echo "Starting scripts"
time python -m pop2vec.evaluation.isolate_income_subset
time python -m pop2vec.evaluation.isolate_marriage_subset

python -m convert_data_to_sqlite.py

# subset embeddings for different models
echo "Extracting subset for llm new"
time python -m pop2vec.evaluation.extract_embedding_subset --model llm_new

echo "Extracting subset for llm old"
time python -m pop2vec.evaluation.convert_embeddings_to_hdf5
time python -m pop2vec.evaluation.extract_embedding_subset --model llm_old 

echo "Extracting subset for network"
time python -m pop2vec.evaluation.convert_pickle_embeddings 
time python -m pop2vec.evaluation.extract_embedding_subset --model network


echo "job ended" 