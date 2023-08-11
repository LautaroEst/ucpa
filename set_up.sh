#!/bin/bash
# Description: Setup the project

project="ucpa"
env_name="ucpa"

echo ""
echo "Starting project..."
date
echo ""

source ~/.initproject $project $env_name
pip install -e .

# Set up dataset
DATA_DIR=$(pwd)/data
datasets=("$project"_trec "$project"_sst2 "$project"_agnews "$project"_dbpedia)
mkdir -p $DATA_DIR
wget https://github.com/tonyzhaozh/few-shot-learning/archive/refs/heads/main.zip -O $MAIN_DATA_DIR/tonyzhao-few-shot-learning.zip
for dataset in ${datasets[@]}; do
    IFS='_' read -ra ADDR <<< $dataset
    dataset_basename=${ADDR[1]}
    echo Processing $dataset_basename...
    if [ ! -d $MAIN_DATA_DIR/$dataset ]; then
        mkdir -p $MAIN_DATA_DIR/temp
        unzip $MAIN_DATA_DIR/tonyzhao-few-shot-learning.zip "few-shot-learning-main/data/$dataset_basename/*" -d $MAIN_DATA_DIR/temp
        mkdir -p $MAIN_DATA_DIR/$dataset
        mv $MAIN_DATA_DIR/temp/few-shot-learning-main/data/$dataset_basename/* $MAIN_DATA_DIR/$dataset
        rm -rf $MAIN_DATA_DIR/temp
    fi
    ln -s $MAIN_DATA_DIR/$dataset/ $DATA_DIR/$dataset
done
rm $MAIN_DATA_DIR/tonyzhao-few-shot-learning.zip
echo Finished setting up datasets.

# Set up checkpoints
echo Setting up checkpoints...
CHECKPOINT_DIR=$(pwd)/checkpoints
SCRIPTS_DIR=$(pwd)/scripts

mkdir -p $CHECKPOINT_DIR
echo $MAIN_MODELS_DIR

python $SCRIPTS_DIR/download_checkpoints.py \
        --results-dir $MAIN_MODELS_DIR

for model in $MAIN_MODELS_DIR/*; do
    ln -s $model $CHECKPOINT_DIR
done
echo Finished setting up checkpoints.

# Set up results
echo Setting up results directory...
RESULTS_DIR=$(pwd)/results
mkdir -p $RESULTS_DIR
echo Finished setting up results directory.

# Set up logs   
echo Setting up logs directory...
LOGS_DIR=$(pwd)/logs
mkdir -p $LOGS_DIR
echo Finished setting up logs directory.

echo Done setting up project.

