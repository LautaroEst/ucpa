#!/bin/bash
#
#$ -S /bin/bash
#$ -N test
#$ -o /homes/eva/q/qestienne/projects/explainability/logs/test_out.log
#$ -e /homes/eva/q/qestienne/projects/explainability/logs/test_err.log
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

# Configure test
TESTS_DIR=./scripts/python/tests
TEST_NAME=$1

# Run the experiment
python $TESTS_DIR/$TEST_NAME.py --root_directory=.