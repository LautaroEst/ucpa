# UCPA Experiments

## Installation and configuration

Create a virtual environment and activate it:
```
conda create -n ucpa python=3.10
conda activate ucpa
```

Clone the repository and install required packages:
```
git clone https://https://github.com/LautaroEst/ucpa.git
cd ucpa
pip install -r requirements.txt
```

Install this package in editable mode
```
pip install -e .
```

Download the models checkpoints running the following script from the main directory:
```
bash download_checkpoints.sh
```

## Usage

Run the experiments:
```
bash run_experiments.sh $EXPERIMENT_NAME $EXPERIMENT_CONFIG $INPUT_FILES
```