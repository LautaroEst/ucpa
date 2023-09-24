#!/bin/bash

project="interpretability"
env_name="fair"

source ~/.bashrc
conda activate $env_name

# Make link to data
# (Data was downloaded from tonyzhao/few-shot)
if [ ! -d data ]; then
    mkdir data
    ln -s $DATA_DIR/ucpa_trec data/trec
    ln -s $DATA_DIR/ucpa_sst2 data/sst2
    ln -s $DATA_DIR/ucpa_agnews data/agnews
    ln -s $DATA_DIR/ucpa_dbpedia data/dbpedia
fi
conda deactivate