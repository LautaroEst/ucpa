#!/bin/bash
#
#$ -S /bin/bash
#$ -N paper_results
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/paper_results_out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/paper_results_err.log
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
declare -a metrics=("norm_cross_entropy" "error_rate")
declare -a datasets=(
    # "tony_zhao_trec"
    "tony_zhao_sst2"
    # "tony_zhao_agnews"
    # "tony_zhao_dbpedia"
)
declare -a models=(
    "gpt2-xl"
    # "t5-small"
    # "google--flan-t5-small"
)

# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         for seed in "${seeds[@]}"; do

#             # Echo the experiment configuration
#             echo ">>> Running experiment: $script_name - $dataset - $model - $seed <<<"

#             # Create the results directory
#             mkdir -p results/$script_name/$dataset/$model/$seed

#             # Run datasets on the model
#             python scripts/python/few_shot.py \
#                 --root_directory=. \
#                 --experiment_name=$script_name \
#                 --model=$model \
#                 --dataset=$dataset \
#                 --config=configs/$script_name/${model}_${dataset}.jsonl \
#                 --seed=$seed

#             # Run calibration on predictions
#             python scripts/python/calibrate_predictions.py \
#                 --root_directory=. \
#                 --experiment_name=$script_name \
#                 --model=$model \
#                 --dataset=$dataset \
#                 --config=configs/$script_name/${model}_${dataset}.jsonl \
#                 --seed=$seed
#         done
#     done
# done

# Plot results
python scripts/python/plot_paper_results.py \
    --root_directory=. \
    --experiment_name=$script_name \
    --models="${models[*]// / }" \
    --datasets="${datasets[*]// / }" \
    --metrics="${metrics[*]// / }" \
    --seeds="${seeds[*]// / }" \
    --bootstrap=100