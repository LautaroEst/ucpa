
import argparse
import json
import numpy as np
import glob






def read_config(config_file):
    """ Read config file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def save_config(config, config_file):
    """ Save config file."""
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)


def seeds_generator(seed,n_seeds=10):
    """ Generate seeds."""
    rs = np.random.RandomState(seed)
    seeds = rs.randint(0, 10000, size=n_seeds)
    for s in seeds:
        yield s