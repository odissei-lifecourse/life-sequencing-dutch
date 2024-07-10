#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=240G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/create_json_seq_stderr.txt
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/create_json_seq_stdout.txt

echo "job started"

cd /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/ 

date
time python -m src.new_code.create_life_seq_jsons projects/dutch_real/create_json_seq_cfg.json

echo "job ended successfully"