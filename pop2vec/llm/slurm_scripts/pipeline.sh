#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4 
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --mem=20G
#SBATCH -p fat_rome
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out


# function to check if $2 is in $1
stringContain() { case $2 in *$1* ) return 0;; *) return 1;; esac ;}

# This function can be called with `source script.sh` -> fast initialization during interactive jobs
initialize() {
	# note that `declare` are local variables and will not be available out of function scope

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

	source $REPO_DIR/requirements/2023_snel_modules.sh	
	source "$VENV/bin/activate" 
	
	echo "job started" 
	cd $REPO_DIR/src/llm/
}


if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    initialize
else
    initialize

    date
    srun python -m pop2vec.llm.src.new_code.pipeline pop2vec/llm/projects/dutch_real/pipeline_cfg.json

    echo "job ended"
fi

