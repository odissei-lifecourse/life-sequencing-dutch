#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH -p gpu
#SBATCH --gpus-per-node=2
#SBATCH -t 00:30:00
#SBATCH -e %x-%j.err 
#SBATCH -o %x-%j.out 

export ROOT_DIR=/home/benjamic
export REPO_DIR=$ROOT_DIR/life-sequencing-dutch
export VENV=$REPO_DIR/.venv/bin/activate

#load the modules
source $REPO_DIR/src/llm/slurm_scripts/2023_snel_modules.sh

#source the virtual environment
source $VENV

# Move to location
cd $REPO_DIR/src/llm/

#Start training
python -m src.new_code.pretrain --accelerator gpu --ddpstrategy gloo --device $SLURM_GPUS_ON_NODE --config projects/dutch_real/pretrain_cfg.json

