#!/bin/bash
#
#$ -S /bin/bash
#$ -N paper_results2
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/paper_results2_out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/paper_results2_err.log
#$ -q long.q
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
declare -a seeds=(82033 12782 1263 987 12299 9203 4 20343 43 92374)

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
        for seed in "${seeds[@]}"; do

            # Echo the experiment configuration
            echo ">>> Running experiment: $script_name - $dataset - $model - $seed <<<"

            # Create the results directory
            mkdir -p results/$script_name/$dataset/$model/$seed

            # Run datasets on the model
            python scripts/python/few_shot.py \
                --root_directory=. \
                --experiment_name=$script_name \
                --model=$model \
                --dataset=$dataset \
                --config=configs/$script_name/${model}_${dataset}.jsonl \
                --seed=$seed

            # Run calibration on predictions
            python scripts/python/calibrate_prediction.py \
                --root_directory=. \
                --experiment_name=$script_name \
                --model=$model \
                --dataset=$dataset \
                --config=configs/$script_name/${model}_${dataset}.jsonl \
                --seed=$seed
        done
    done
done