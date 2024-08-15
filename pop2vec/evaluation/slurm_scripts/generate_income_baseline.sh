#!/bin/bash
#
#SBATCH --job-name=generate_income_baseline
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=50G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err

echo "job started" 

cd /gpfs/ostor/ossc9424/homedir/

module purge 
source ossc_env/bin/activate
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-intel-2022a
module load matplotlib/3.5.2-foss-2022a

cd /gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/

date
time python -m pop2vec.evaluation.generate_income_baseline

echo "job ended" 