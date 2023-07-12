#! /bin/bash -ex
# Description: Run an experiment with the given name and config file

# Set the paths
SCRIPTS_DIR=$(pwd)/scripts
DATA_DIR=$(pwd)/data
RESULTS_DIR=$(pwd)/results
CONFIGS_DIR=$(pwd)/configs

# Read command line arguments
EXPERIMENT_NAME=$1
CONFIG_FILE=$2

# Create the results directory
mkdir -p $RESULTS_DIR

# Run the experiment
python $SCRIPTS_DIR/$EXPERIMENT_NAME.py \
    --data-dir $DATA_DIR \
    --results-dir $RESULTS_DIR \
    --config-file $CONFIGS_DIR/$EXPERIMENT_NAME/$CONFIG_FILE.json