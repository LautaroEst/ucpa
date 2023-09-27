

import argparse
import json
import os
import numpy as np


def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs_dicts = [json.loads(config) for config in f.read().splitlines()]
    setattr(args,"configs_dicts",configs_dicts)
    
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()
    root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.dataset, args.model, str(args.seed))

    for config in args.configs_dicts:

        # Results path for this config
        os.makedirs(os.path.join(root_save_path,str(config["id"])), exist_ok=True)

        # Load results and calibrate
        train_logits = np.load(os.path.join(root_save_path,str(config["id"]),f"train.logits.npy"))
        train_labels = np.load(os.path.join(root_save_path,str(config["id"]),f"train.labels.npy"))
        test_logits = np.load(os.path.join(root_save_path,str(config["id"]),f"test.logits.npy"))
        test_labels = np.load(os.path.join(root_save_path,str(config["id"]),f"test.labels.npy"))
        for method in config["calibration_methods"]:
            for n_samples in config["num_calibration_train_samples"]:
                cal_logits = run_calibration(train_logits, train_labels, test_logits, test_labels, method=method)
                


if __name__ == "__main__":
    main()