#!/bin/bash
#
#SBATCH --job-name=spreadsheets
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -p work_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/synthetic/logs/generate_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/synthetic/logs/generate_stdout.txt

module purge 
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source /gpfs/ostor/ossc9424/homedir/ossc_new/bin/activate 

DATAPATH="/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/"

echo "job started" 
cd /gpfs/ostor/ossc9424/homedir/synthetic_data/

date
time python gen_spreadsheets.py "$DATAPATH/dutch_real/raw_data/" "$DATAPATH/synthetic/raw_data/"
time python gen_spreadsheets.py "$DATAPATH/dutch_real/data/" "$DATAPATH/synthetic/data/"

echo "job ended successfully" 