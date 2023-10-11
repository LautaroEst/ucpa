#!/bin/bash
#
#$ -S /bin/bash
#$ -N predict_with_model
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/predict_with_model_out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/predict_with_model_err.log
#$ -q long.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=48G,ram_free=64G,mem_free=30G
#

# Configure environment
project_name="ucpa"
env_name="ucpa"
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project_name
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)

script_name="plot"
declare -a seeds=(82033 12782 1263 987 12299)
config="samples_shots"

# Echo the experiment configuration
echo ">>> Running $script_name script <<<"

# Create the results directory
mkdir -p results/$script_name/

# # Run datasets on the model
python scripts/python/plot.py \
    --root_directory=. \
    --experiment_name=$script_name \
    --config="configs/$script_name/$config.json" \
    --seed="${seeds[*]// / }"