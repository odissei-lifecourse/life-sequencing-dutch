#!/bin/bash
#
#SBATCH --job-name=create_fake_data
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:02:00
#SBATCH --mem=100MB
#SBATCH -p rome 
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out

stringContain() { case $2 in *$1* ) return 0;; *) return 1;; esac ;}

if stringContain "ossc" $USER; then
    declare DATADIR="/gpfs/ostor/ossc9424/homedir/data"
else
    declare DATADIR="/projects/0/prjs1019"
fi 

# add more users here as necessary
if stringContain "ossc" $USER; then
    declare ROOTDIR="/gpfs/ostor/ossc9424/homedir"
    declare VENV="$ROOTDIR/ossc_env"
elif stringContain "fhafner" $USER; then 
    declare ROOTDIR="/gpfs/home4/$USER"
    declare REPO_DIR="$ROOTDIR/repositories/life-sequencing-dutch"
    declare VENV="$REPO_DIR/.venv"
fi

module purge 
module load 2022 
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
source "$VENV/bin/activate" 

# DATAPATH="/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/"

echo "job started" 
cd $REPO_DIR

date
time python -m src.others.synthetic_data_generation.create_fake_data \
 --cfg src/others/synthetic_data_generation/fake_data_cfg.json \
 --dry-run

echo "job ended" 
