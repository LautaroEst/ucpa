#!/bin/bash
# Description: Run an experiment with the given name and config file
#
#$ -S /bin/bash
#$ -N ucpa
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/err.log
#$ -q all.q@svatava.q,all.q@@blade
#

# Project and environment names
project="ucpa"
env_name="ucpa"

# Models to run
models_names=("gpt2" "t5-small")

source ~/.initproject $project $env_name

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