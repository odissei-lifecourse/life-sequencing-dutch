#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -p gpu 
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
	
	module purge 
	module load 2022 
	module load Python/3.10.4-GCCcore-11.3.0
        #module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
	module load SciPy-bundle/2022.05-foss-2022a
        module load h5py/3.7.0-foss-2022a
	module load matplotlib/3.5.2-foss-2022a
	source "$VENV/bin/activate" 
	
	cd $REPO_DIR/src/llm/
        export CUDA_VISIBLE_DEVICES=0
       

        echo "job started" 
}


main() {
	date
	python -m src.new_code.pretrain projects/dutch_real/pretrain_cfg.json
	# for debugging on GPU
	#CUDA_LAUNCH_BLOCKING=1 python -m src.new_code.pretrain projects/dutch_real/pretrain_cfg.json
        # for debugging on CPU 
	#python -m src.new_code.pretrain projects/dutch_real/pretrain_cfg.json	
	echo "job ended"
}



if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    initialize
else
    initialize
    main
fi



#	cd "$HOMEDIR"/Tanzir/LifeToVec_Nov/





