#!/bin/bash
#SBATCH --job-name=align_0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=ossc9424vm1
#SBATCH --time=0:20:00
#SBATCH --mem=200G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/oss9424tpal/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/oss9424tpal/logs/%j.%x.out

echo "job started"
date 



# ---------- modules / env ---------------------------------------------
source requirements/load_venv.sh

# ---------- run --------------------------------------------------------
CFG="pop2vec/graph/config/OSSC/align_0.cfg"


time python -m pop2vec.graph.src.align_embedding_spaces \
         --config "$CFG" \
         --fraction 0.1 \

echo "job ended successfully"
