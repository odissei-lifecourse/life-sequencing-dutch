#!/bin/bash
#
#SBATCH --job-name=eval_set_report 
#SBATCH --n_tasks_per_node=8
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH -p work_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/%x.%j.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/%x.%j.out

# Runs the reports for LLM and Network
# Jobs are ordered in increasing empirical duration

initialize() { 
    cd /gpfs/ostor/ossc9424/homedir/

    module purge 
    source ossc_env/bin/activate 
    module load 2022 
    module load Python/3.10.4-GCCcore-11.3.0
    module load matplotlib/3.5.2-foss-2022a

    cd /gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation

}

main () {
    date 

    NONLINEAR_MODEL="support_vector_machine" 
    LINEAR_MODEL="ridge_regression"


    python -m pop2vec.evaluation.write_embedding_metadata --collection_name llm_eval_set 
    python -m pop2vec.evaluation.write_embedding_metadata --collection_name network_eval_set

    echo "Running llm with ${LINEAR_MODEL}" 
    python -m pop2vec.evaluation.generate_life_course_report \
        --collection_name llm_eval_set \
        --pred_model $LINEAR_MODEL \
        --savename "results/llm_${LINEAR_MODEL}_" 
    
    echo "Running network with ${LINEAR_MODEL}" 
    python -m pop2vec.evaluation.generate_life_course_report \
        --collection_name network_eval_set \
        --pred_model $LINEAR_MODEL \
        --savename "results/net_${LINEAR_MODEL}_" 
    
    echo "Running network with ${NONLINEAR_MODEL}" 
    python -m pop2vec.evaluation.generate_life_course_report \
        --collection_name network_eval_set \
        --pred_model $NONLINEAR_MODEL \
        --savename "results/net_${NONLINEAR_MODEL}_" 
    
    echo "Running llm with ${NONLINEAR_MODEL}" 
    python -m pop2vec.evaluation.generate_life_course_report \
        --collection_name llm_eval_set \
        --pred_model $NONLINEAR_MODEL \
        --savename "results/llm_${NONLINEAR_MODEL}_" 

    echo "End of script" 
    exit     
}

if [[ "${[BASH_SOURCE[0]]}" != "${0}" ]]; then 
    initialize
else 
    initialize
    main 
fi 
