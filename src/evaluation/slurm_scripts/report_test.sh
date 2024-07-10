#!/bin/bash
#
#SBATCH --job-name=report_test
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err

module purge 
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0

echo "job started" 
cd /gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/

date
time python report_tests.py

echo "job ended successfully" 