#!/bin/bash -ex
# Description: Run an experiment with the given name and config file
#
#$ -S /bin/bash
#$ -N ucpa
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/err.log
#$ -q all.q@svatava.q,all.q@@blade
#

project="ucpa"
env_name="ucpa"
source ~/.initproject $project $env_name

./run_experiment.sh zero_shot gpt2
./run_experiment.sh zero_shot t5-small