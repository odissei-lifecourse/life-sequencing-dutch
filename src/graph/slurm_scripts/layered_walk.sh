#!/bin/bash
#
#SBATCH --job-name=layered_walk
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/layered_walk_stdout.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/layered_walk_stdout.txt

module purge 
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0

echo "job started" 
cd /gpfs/ostor/ossc9424/homedir/Dakota_network/

date
time python layered_random_walk.py --year 2016 --start_int 5

echo "job ended successfully" 
