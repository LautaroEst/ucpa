#!/bin/bash
#
#$ -S /bin/bash
#$ -N gpt2-xl
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/gpt2-xl_out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/gpt2-xl_err.log
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

script_name="predict_with_model"

model="gpt2-xl"
# model="gpt2"
# model="google--flan-t5-small"
# model="google--flan-t5-xl"
# model="t5-small"
# model="google--flan-t5-xxl"
# model="meta-llama--Llama-2-7b-hf"

# declare -a seeds=(82033 12782 1263 987 12299 9203 4 20343 43 92374)
declare -a seeds=(82033 12782 1263 987 12299)

# Echo the experiment configuration
echo ">>> Running $script_name script for model $model <<<"

# Create the results directory
mkdir -p results/$script_name/$model/$seed

# # Run datasets on the model
python scripts/python/predict_with_model.py \
    --root_directory=. \
    --experiment_name=$script_name \
    --model=$model \
    --seed="${seeds[*]// / }"