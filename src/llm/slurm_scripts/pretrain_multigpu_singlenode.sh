#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:30:00
#SBATCH -e %x-%j.err 
#SBATCH -o %x-%j.out 

export ROOT_DIR=/home/fhafner/
export REPO_DIR=$ROOT_DIR/repositories/life-sequencing-dutch
export VENV=$REPO_DIR/.venv/bin/activate

#load the modules
source $REPO_DIR/requirements/snel_modules_2023.sh

#source the virtual environment
source $VENV

# Move to location
cd $REPO_DIR/src/llm/

#Start training
srun python -m src.new_code.pretrain \
       --accelerator gpu \
       --ddpstrategy auto \
       --device $SLURM_GPUS_ON_NODE \
       --batch 256 \
       --hparams src/new_code/regular_hparams_large.txt \
       --config projects/dutch_real/pretrain_cfg.json

