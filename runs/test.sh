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

# Tests to run
# tests_names=("prompt_template" "gpt2_model")
tests_names=("gpt2_model" )


source ~/.initproject $project $env_name

echo ""
date
echo "Test scripts"

for test_name in "${tests_names[@]}"
do
    echo ""
    echo "Starting $test_name test..."
    echo ""
    ./run_test.sh $test_name
done
