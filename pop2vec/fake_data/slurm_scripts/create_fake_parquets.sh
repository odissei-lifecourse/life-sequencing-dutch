#!/bin/bash
#
#SBATCH --job-name=fake_parquet
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10 
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --mem=50G
#SBATCH -p rome
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

# NOTE: this currently supersedes the scripts `create_fake_data.sh`.

echo "job started"

source requirements/load_venv.sh

python -m pop2vec.fake_data.generate_rinpersoon /projects/0/prjs1019/data/fake_data_v0/fake_rinpersoons.csv

python -m pop2vec.fake_data.generate_step2_data








