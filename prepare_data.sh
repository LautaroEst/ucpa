#!/bin/bash

project="ucpa"
env_name="ucpa"

source ~/.bashrc
conda activate $env_name

# Make link to data
# (Data was downloaded from tonyzhao/few-shot)
if [ ! -d data ]; then
    mkdir -p data/tony_zhao
    ln -s $DATA_DIR/ucpa_trec data/tony_zhao/trec
    ln -s $DATA_DIR/ucpa_sst2 data/tony_zhao/sst2
    ln -s $DATA_DIR/ucpa_agnews data/tony_zhao/agnews
    ln -s $DATA_DIR/ucpa_dbpedia data/tony_zhao/dbpedia
fi
conda deactivate