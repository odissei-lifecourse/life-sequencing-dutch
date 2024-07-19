#!/bin/bash
# This is how the venv on snellius (OSSC and regular) should be activated

declare ENV_NAME="ossc_env_may2"

module purge 
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
source ${ENV_NAME}/bin/activate