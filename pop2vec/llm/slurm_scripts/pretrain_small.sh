#!/bin/bash
#
#SBATCH --job-name=pretrain_small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --mem=80G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/dutch_real/logs/%x.%j.out

echo "job started"
declare PREFIX="/gpfs/ostor/ossc9424/homedir/"
declare VENV="ossc_new"

module purge 
module load 2022 
module load Python/3.10.4-GCCCore-11.3.0
module load PyTorch/1.12.0-foss-CUDA-11.7.0
module load SciPy-bundle/2022.05-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source "$HOMEDIR"/"$VENV"/bin/activate

export CUDA_VISIBLE_DEVICES=0

cd "$PREFIX"/Tanzir/LifeToVec_Nov/

date
time python -m src.new_code.pretrain projects/dutch_real/pretrain_cfg_small.json

echo "job ended successfully"