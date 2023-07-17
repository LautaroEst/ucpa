#! /bin/bash -ex
# Description: Download checkpoints from Huggingface

CHECKPOINT_DIR=$(pwd)/checkpoints
SCRIPTS_DIR=$(pwd)/scripts

# mkdir -p $CHECKPOINT_DIR
ln -s /mnt/extra/lautaro/checkpoints $CHECKPOINT_DIR

python $SCRIPTS_DIR/download_checkpoints.py \
        --results-dir $CHECKPOINT_DIR