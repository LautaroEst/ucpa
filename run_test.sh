#!/bin/bash -ex
# Description: Run test from tests directory

# Set the paths
TESTS_DIR=$(pwd)/tests
DATA_DIR=$(pwd)/data
CHECKPOINTS_DIR=$(pwd)/checkpoints

# Read command line arguments
TEST_NAME=$1

# Run the experiment
python $TESTS_DIR/$TEST_NAME.py \
    --data-dir $DATA_DIR \
    --checkpoints-dir $CHECKPOINTS_DIR