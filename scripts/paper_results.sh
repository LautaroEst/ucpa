#!/bin/bash
#
#$ -S /bin/bash
#$ -N paper_results
#$ -o /homes/eva/q/qestienne/projects/explainability/logs/paper_results_out.log
#$ -e /homes/eva/q/qestienne/projects/explainability/logs/paper_results_err.log
#$ -q all.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=16G,ram_free=64G,mem_free=30G
#

# Configure environment
project_name="ucpa"
env_name="ucpa"
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project_name
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)


script_name="paper_results"
seed=82033

declare -a dataset=(
    "tony_zhao_trec"
    # "tony_zhao_sst2"
    # "tony_zhao_agnews"
    # "tony_zhao_dbpedia"
)

declare -a models=(
    "gpt2"
    # "t5-small"
    # "google--flan-t5-small"
)

for model in "${models[@]}"; do
    for dataset in "${dataset[@]}"; do

        # Echo the experiment configuration
        echo ">>> Running experiment: $script_name - $dataset - $model"

        # Create the results directory
        mkdir -p results/$script_name/$dataset/$model/$seed

        # Run the experiment
        python scripts/python/few_shot.py \
            --root_directory=. \
            --experiment_name=$script_name \
            --model=$model \
            --dataset=$dataset \
            --config=configs/$script_name/${model}_${dataset}.jsonl \
            --seed=$seed
    done
done