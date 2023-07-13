
import argparse
import json
import numpy as np
import glob



def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="./results"
    )
    parser.add_argument(
        "--config-file", 
        type=str
    )
    parser.add_argument(
        "--input-files",
        type=str,
        default=""
    )
    args = parser.parse_args()
    args.input_files = glob.glob(args.input_files)
    return args


def read_config(config_file):
    """ Read config file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def seeds_generator(seed,n_seeds=10):
    """ Generate seeds."""
    rs = np.random.RandomState(seed)
    seeds = rs.randint(0, 10000, size=n_seeds)
    for s in seeds:
        yield s