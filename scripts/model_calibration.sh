#!/bin/bash
#
#$ -S /bin/bash
#$ -N model_calibration
#$ -o /mnt/matylda3/qestienne/projects/ucpa/logs/model_calibration_out.log
#$ -e /mnt/matylda3/qestienne/projects/ucpa/logs/model_calibration_err.log
#$ -q long.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=16G,ram_free=64G,mem_free=30G
#

# Configure environment
project_name="ucpa"
env_name="ucpa"
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project_name
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)

script_name="model_calibration"

model="gpt2-xl"
# model="gpt2"
# model="t5-small"
# model="google--flan-t5-xxl"
# model="meta-llama--Llama-2-7b-hf"

declare -a seeds=(82033 12782 1263 987 12299 9203 4 20343 43 92374)
declare -a metrics=("norm_cross_entropy" "error_rate")
declare -a calibration_methods=("UCPA" "SUCPA" "UCPA-naive" "SUCPA-naive" "affine_bias_only")
declare -a num_calibration_train_samples=(10 20 40 80 200 400 600)
num_train_samples=600
num_test_samples=-1
bootstrap=100

for seed in "${seeds[@]}"; do

    # Echo the experiment configuration
    echo ">>> Running experiment: $script_name - $model - $seed <<<"

    # Create the results directory
    mkdir -p results/$script_name/$model/$seed

    # Run datasets on the model
    python scripts/python/few_shot.py \
        --root_directory=. \
        --experiment_name=$script_name \
        --model=$model \
        --num_train_samples=$num_train_samples \
        --num_test_samples=$num_test_samples \
        --seed=$seed
        
    # Run calibration on predictions
    python scripts/python/calibrate_predictions.py \
        --root_directory=. \
        --experiment_name=$script_name \
        --model=$model \
        --calibration_methods="${calibration_methods[*]// / }" \
        --num_calibration_train_samples="${num_calibration_train_samples[*]// / }" \
        --seed=$seed
done

# Plot results
python scripts/python/plot_model_calibration_results.py \
    --root_directory=. \
    --experiment_name=$script_name \
    --model=$model \
    --metrics="${metrics[*]// / }" \
    --seeds="${seeds[*]// / }" \
    --bootstrap=$bootstrap