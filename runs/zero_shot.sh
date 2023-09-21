#!/bin/bash
# Description: Run an experiment with the given name and config file
#
#$ -S /bin/bash
#$ -N ucpa
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/err.log
#$ -q all.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=32G,ram_free=64G,mem_free=16G
#

# Project and environment names
project="ucpa"
env_name="ucpa"

# Init project
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)

# Models to run
# models_names=("gpt2" "t5-small")
# models_names=("gpt2-xl" "google/flan-t5-small")
models_names=("meta-llama/Llama-2-7b-hf" )

echo ""
date
echo "Running models in zero-shot mode..."

for model_name in "${models_names[@]}"
do
    echo ""
    echo "Experiment on $model_name model..."
    echo ""
    ./run_experiment.sh zero_shot $model_name
done
