#!/bin/bash
#SBATCH --job-name=216
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --time=120:00:00
#SBATCH --mem=400G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/logs/%j.%x.out

n_gpus=4

echo "job started"
date

source requirements/load_venv.sh
# export CUDA_VISIBLE_DEVICES=0

# ---------- run ----------
cfg="/Users/tanzir5/Documents/GitHub/life-sequencing-dutch/generated_configs_ft/liss_filtered/BASE-event/lr3e-05_bs8.json"

date
export NCCL_SOCKET_IFNAME=ib0

srun python -m pop2vec.evaluation.prediction_settings.finetune_runner "$cfg"

date
echo "job ended successfully"
