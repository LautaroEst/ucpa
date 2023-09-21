#!/bin/bash
#
#$ -S /bin/bash
#$ -N zero_shot
#$ -o /homes/eva/q/qestienne/projects/explainability/logs/zero_shot_out.log
#$ -e /homes/eva/q/qestienne/projects/explainability/logs/zero_shot_err.log
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


script_name="zero_shot"
seed=82033

declare -a dataset=(
    "trec"
    "sst2"
    "agnews"
    "dbpedia"
)

declare -a models=(
    "gpt2"
    "t5-small"
    "google--flan-t5-small"
)

for model in "${models[@]}"; do
    for dataset in "${dataset[@]}"; do

        # Echo the experiment configuration
        echo ">>> Running experiment: $script_name - $dataset - $model"

        # Create the results directory
        mkdir -p results/$script_name/$dataset/$model/$seed

        # Run the experiment
        python scripts/python/$script_name.py \
            --root_directory=. \
            --base_model=$model \
            --dataset=$dataset \
            --config-file=configs/$script_name/${model}_${dataset}.json \
            --seed=$seed
    done
done