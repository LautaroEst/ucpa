#!/bin/bash
#
#$ -S /bin/bash
#$ -N model_calibration
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/model_calibration_out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/model_calibration_err.log
#$ -q long.q
#$ -l matylda3=0.5,ram_free=64G,mem_free=30G
#

# Configure environment
project_name="ucpa"
env_name="ucpa"
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project_name

script_name="plot_models_results"
declare -a models=(
    "gpt2"
    "gpt2-xl"
    "google--flan-t5-small"
    "google--flan-t5-xl"
    "t5-small"
    "meta-llama--Llama-2-7b-hf"
)
declare -a seeds=(82033 12782 1263 987 12299)
declare -a calibration_methods=("UCPA" "SUCPA" "UCPA-naive" "SUCPA-naive" "affine_bias_only")
metric="error_rate"
num_shots=0

mkdir -p results/$script_name
python scripts/python/plot_all_models.py \
    --root_directory=. \
    --experiment_name=$script_name \
    --model="${models[*]// / }" \
    --metric=$metric \
    --num_shots=$num_shots \
    --seeds="${seeds[*]// / }"