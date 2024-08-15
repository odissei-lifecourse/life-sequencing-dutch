#!/bin/bash
#
#SBATCH --job-name=create_fake_data
#SBATCH --ntasks-per-node=50
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=5GB
#SBATCH -p rome 
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

	
	# DATAPATH="/gpfs/ostor/ossc9424/homedir/Tanzir/LifeToVec_Nov/projects/"
	
	echo "job started" 
	cd $REPO_DIR
	source requirements/snel_modules_2023.sh	
	source .venv/bin/activate

}


if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    initialize
else
    initialize
   
    date
    time python -m pop2vec.fake_data.create_fake_data \
    	--cfg pop2vec/fake_data/configs/llm_data.json 

	time python -m pop2vec.fake_data.create_fake_data \
		--cfg pop2vec/fake_data/configs/original_data.json

    echo "job ended" 
fi


