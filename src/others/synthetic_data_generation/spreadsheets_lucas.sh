#!/bin/bash
#
#SBATCH --job-name=spreadsheets_lucas
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -p work_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/synthetic_data/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/synthetic_data/logs/%x.%j.out

module purge 
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
source /gpfs/ostor/ossc9424/homedir/ossc_env/bin/activate 

DATAPATH="/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/"

echo "job started" 
DATAPATH="/gpfs/ostor/ossc9424/homedir/"

cd /gpfs/ostor/ossc9424/homedir/synthetic_data/

date
time python gen_spreadsheets.py "$DATAPATH/cbs_data/real/" "$DATAPATH/cbs_data/synthetic/" "100000" csv
time python gen_spreadsheets.py "$DATAPATH/cbs_data/real/" "$DATAPATH/cbs_data/synthetic/" "100000" sav

python gen_csv_from_jsons.py "$DATAPATH/cbs_data/synthetic/json" "$DATAPATH/cbs_data/synthetic/final/"

echo "job ended successfully" 
