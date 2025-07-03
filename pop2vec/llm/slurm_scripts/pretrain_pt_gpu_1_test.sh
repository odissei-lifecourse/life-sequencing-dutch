#!/bin/bash
#SBATCH --job-name=pretrain_pt_gpu_1_test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH -e /home/tislampial/logs/%x.%j.err
#SBATCH -o /home/tislampial/logs/%x.%j.out

echo "job started"

# ---------- paths ------------------------------------------------------
PROJECT_DIR="/home/tislampial/life-sequencing-dutch"
cd "$PROJECT_DIR"


# ---------- modules / env ---------------------------------------------
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a
module load SciPy-bundle/2022.05-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source requirements/load_venv.sh 


# ---------- run --------------------------------------------------------
CFG="pop2vec/llm/configs/Snellius/pretrain_pt_gpu_1_test.cfg"

date
time python -m pop2vec.llm.src.new_code.pytorch_port.pretrain \
         --config "$CFG" \
         --val_check_interval 0.5 \
         --log_every 10 \
         --strategy single_gpu

echo "job ended successfully"
