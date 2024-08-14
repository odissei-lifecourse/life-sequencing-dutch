#!/bin/bash
# 
#SBATCH --job-name=test_cpu_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH -p work_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.out


echo "job started"
HOMEDIR="/gpfs/ostor/ossc9424/homedir/"

cd "$HOMEDIR"/Tanzir/LifeToVec_Nov

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source "$HOMEDIR"/ossc_env_may2/bin/activate

date
echo "starting python script" 
srun python test_parallel_cpu.py

echo "job ended" 