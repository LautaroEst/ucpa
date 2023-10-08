

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from ucpa.calibration import AffineCalibrator, UCPACalibrator

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--calibration_methods", type=str, default="")
    parser.add_argument("--num_calibration_train_samples", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.root_directory,f"configs/{args.experiment_name}/{args.model}.jsonl"), "r") as f:
        configs_dicts = [json.loads(config) for config in f.read().splitlines()]
    setattr(args,"configs_dicts",configs_dicts)
    
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()
    root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.model, str(args.seed))
    calibration_methods = args.calibration_methods.split(" ")
    num_calibration_train_samples = [int(s) for s in args.num_calibration_train_samples.split(" ")]

    for config in args.configs_dicts:

        # Results path for this config
        os.makedirs(os.path.join(root_save_path,str(config["id"]),"calibration"), exist_ok=True)

        # Random state
        rs = np.random.RandomState(args.seed+config["id-num"])

        # Load results and calibrate
        train_logits = np.load(os.path.join(root_save_path,str(config["id"]),f"train.logits.npy"))
        train_labels = np.load(os.path.join(root_save_path,str(config["id"]),f"train.labels.npy"))
        test_logits = np.load(os.path.join(root_save_path,str(config["id"]),f"test.logits.npy"))
        test_labels = np.load(os.path.join(root_save_path,str(config["id"]),f"test.labels.npy"))        
        
        # Train samples:
        train_samples = tqdm(num_calibration_train_samples)
        for n_samples in train_samples:
            sub_train_logits, sub_train_labels = subsample_logits_and_labels(train_logits, train_labels, n_samples, rs=rs)
            for method in calibration_methods:
                train_samples.set_description(desc=f"{method} ({n_samples} samples): ")
                cal_test_logits = run_calibration(sub_train_logits, sub_train_labels, test_logits, method=method)
                np.save(os.path.join(root_save_path,str(config["id"]),"calibration",f"test.{method}.{n_samples}.npy"),cal_test_logits)

                
def subsample_logits_and_labels(logits, labels, n_samples, rs):
    idx = rs.choice(len(labels), n_samples, replace=False)
    new_logits = logits[idx,:].copy()
    new_labels = labels[idx].copy()
    return new_logits, new_labels


def run_calibration(train_logits, train_labels, test_logits, method="UCPA"):

    num_samples, num_classes = train_logits.shape

    train_logits = torch.from_numpy(train_logits)
    train_labels = torch.from_numpy(train_labels)
    test_logits = torch.from_numpy(test_logits)
    
    if method == "affine_bias_only":
        model = AffineCalibrator(num_classes, "bias only", "log-loss")
        model.train_calibrator(train_logits, train_labels)
        cal_logits = model.calibrate(test_logits)
    elif method == "UCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1, tolerance=1e-6)
        model.train_calibrator(train_logits)
        cal_logits = model.calibrate(test_logits)
    elif method == "SUCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1, tolerance=1e-6)
        priors = torch.bincount(train_labels, minlength=num_classes) / num_samples
        model.train_calibrator(train_logits, priors)
        cal_logits = model.calibrate(test_logits)
    elif method == "UCPA":
        model = UCPACalibrator(num_classes, max_iters=20, tolerance=1e-6)
        model.train_calibrator(train_logits)
        cal_logits = model.calibrate(test_logits)
    elif method == "SUCPA":
        model = UCPACalibrator(num_classes, max_iters=20, tolerance=1e-6)
        priors = torch.bincount(train_labels, minlength=num_classes) / num_samples
        model.train_calibrator(train_logits, priors)
        cal_logits = model.calibrate(test_logits)
    else:
        raise ValueError(f"Calibration method {method} not supported.")
    
    cal_logits = cal_logits.numpy()
    return cal_logits        

if __name__ == "__main__":
    main()