#!/bin/bash
#SBATCH --job-name=14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=ossc9424vm1
#SBATCH --time=120:00:00
#SBATCH --mem=50G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/logs/%j.%x.out

n_gpus=1

echo "job started"
date

source requirements/load_venv.sh
export CUDA_VISIBLE_DEVICES=2

# ---------- run ----------
cfg="generated_configs/liss_filtered/medium-random/lr1e-03_bs128.json"

date
export NCCL_SOCKET_IFNAME=ib0

srun python -m pop2vec.evaluation.prediction_settings.train_simple "$cfg"

date
echo "job ended successfully"
