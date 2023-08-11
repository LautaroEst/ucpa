# Set the paths
SCRIPTS_DIR=$(pwd)/scripts
DATA_DIR=$(pwd)/data
ROOT_RESULTS_DIR=$(pwd)/results
CONFIGS_DIR=$(pwd)/configs
CHECKPOINTS_DIR=$(pwd)/checkpoints

# Read command line arguments
EXPERIMENT_NAME=$1
CONFIG_FILE=$2
INPUT_FILES=$(pwd)/$3

# Create the results directory
mkdir -p $ROOT_RESULTS_DIR/$EXPERIMENT_NAME/$CONFIG_FILE

# Run the experiment
if [ "$#" -eq 3 ]; then
    python $SCRIPTS_DIR/$EXPERIMENT_NAME.py \
        --data-dir $DATA_DIR \
        --results-dir $ROOT_RESULTS_DIR/$EXPERIMENT_NAME/$CONFIG_FILE \
        --config-file $CONFIGS_DIR/$EXPERIMENT_NAME/$CONFIG_FILE.json \
        --input-files $INPUT_FILES \
        --checkpoints-dir $CHECKPOINTS_DIR
else
    python $SCRIPTS_DIR/$EXPERIMENT_NAME.py \
        --data-dir $DATA_DIR \
        --results-dir $ROOT_RESULTS_DIR/$EXPERIMENT_NAME/$CONFIG_FILE \
        --config-file $CONFIGS_DIR/$EXPERIMENT_NAME/$CONFIG_FILE.json \
        --checkpoints-dir $CHECKPOINTS_DIR
fi