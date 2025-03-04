#!/bin/bash
#
#SBATCH --job-name=infer_medium2x
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=100G
#SBATCH -p comp_env
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

declare PREFIX="/gpfs/ostor/ossc9424/homedir/"

export CUDA_VISIBLE_DEVICES=2

echo "job started"

module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load SciPy-bundle/2022.05-foss-2022a
module load matplotlib/3.5.2-foss-2022a

source "$PREFIX"/ossc_new/bin/activate

cd "$PREFIX"/Tanzir/LifeToVec_Nov/

date
time python -m pop2vec.llm.src.new_code.infer_embedding pop2vec/llm/configs/Snellius/infer_cfg_medium2x.json

echo "job ended successfully"
