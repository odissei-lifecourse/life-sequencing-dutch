#!/bin/bash
#
# SBATCH --job-name-trans_spell
# SBATCH --ntasks 1
# SBATCH --cpus-per-task 16
# SBATCH --nodes=1
# SBATCH --time=03:00:00
# SBATCH --mem=60G
# SBATCH -p comp_env
# SBATCH -e logs/%x-$j.err
# SBATCH -o logs/%x-%j.out

cd /gpfs/ostor/ossc9424/homedir/users/flavio/life-sequencing-dutch || exit
source requirements/load_venv.sh

python -m pop2vec.utils.transform_spells
