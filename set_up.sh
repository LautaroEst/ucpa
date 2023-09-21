#!/bin/bash
# Description: Setup the project

# project="ucpa"
# env_name="ucpa"

# source ~/.bashrc
# conda activate $env_name
# cd $PROJECTS_DIR/$project


echo ""
echo "Starting project..."
date
echo ""

# Project paths
PROJECT_DATA_DIR=$(pwd)/data
PROJECT_CHECKPOINTS_DIR=$(pwd)/checkpoints
PROJECT_SCRIPTS_DIR=$(pwd)/scripts
PROJECT_RESULTS_DIR=$(pwd)/results
PROJECT_LOGS_DIR=$(pwd)/logs

# # Set up dataset
# datasets=("$project"_trec "$project"_sst2 "$project"_agnews "$project"_dbpedia)
# mkdir -p $PROJECT_DATA_DIR
# wget https://github.com/tonyzhaozh/few-shot-learning/archive/refs/heads/main.zip -O $DATA_DIR/tonyzhao-few-shot-learning.zip
# for dataset in ${datasets[@]}; do
#     IFS='_' read -ra ADDR <<< $dataset
#     dataset_basename=${ADDR[1]}
#     echo Processing $dataset_basename...
#     if [ ! -d $DATA_DIR/$dataset ]; then
#         mkdir -p $DATA_DIR/temp
#         unzip $DATA_DIR/tonyzhao-few-shot-learning.zip "few-shot-learning-main/data/$dataset_basename/*" -d $DATA_DIR/temp
#         mkdir -p $DATA_DIR/$dataset
#         mv $DATA_DIR/temp/few-shot-learning-main/data/$dataset_basename/* $DATA_DIR/$dataset
#         rm -rf $DATA_DIR/temp
#     fi
#     ln -s $DATA_DIR/$dataset/ $PROJECT_DATA_DIR/$dataset
# done
# rm $DATA_DIR/tonyzhao-few-shot-learning.zip
# echo Finished setting up datasets.

# Set up checkpoints
echo Setting up checkpoints...
mkdir -p $PROJECT_CHECKPOINTS_DIR
python $PROJECT_SCRIPTS_DIR/download_checkpoints.py \
        --results-dir $MODELS_CHECKPOINTS_DIR
# for model in $MODELS_CHECKPOINTS_DIR/*; do
#     ln -s $model $PROJECT_CHECKPOINTS_DIR
# done
echo Finished setting up checkpoints.

# # Set up results
# echo Setting up results directory...
# mkdir -p $PROJECT_RESULTS_DIR
# echo Finished setting up results directory.

# # Set up logs   
# echo Setting up logs directory...
# mkdir -p $PROJECT_LOGS_DIR
# echo Finished setting up logs directory.

echo Done setting up project.

