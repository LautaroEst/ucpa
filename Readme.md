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

Set up the project running the following script from the main directory:
```
chmod +x set_up.sh
./set_up.sh
```

## Usage with SGE

```
qsub runs/myrun.sh
```