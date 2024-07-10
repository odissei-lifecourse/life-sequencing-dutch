#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=900G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.out

echo "job started"

cd /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/ 

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source ossc_env_may2/bin/activate 

date
srun python -m src.new_code.pipeline projects/dutch_real/pipeline_cfg.json

echo "job ended"

