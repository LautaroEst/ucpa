

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
        os.makedirs(os.path.join(root_save_path,str(config["id"]),"calibration"), exist_ok=True)

        # Random state
        rs = np.random.RandomState(args.seed)

        # Load results and calibrate
        train_logits = np.load(os.path.join(root_save_path,str(config["id"]),f"train.logits.npy"))
        train_labels = np.load(os.path.join(root_save_path,str(config["id"]),f"train.labels.npy"))
        test_logits = np.load(os.path.join(root_save_path,str(config["id"]),f"test.logits.npy"))
        test_labels = np.load(os.path.join(root_save_path,str(config["id"]),f"test.labels.npy"))        
        
        for method in config["calibration_methods"]:
            tqdm_num_samples = tqdm(config["num_calibration_train_samples"])
            for n_samples in tqdm_num_samples:
                tqdm_num_samples.set_description(desc=f"{method} ({n_samples} samples): ")
                sub_train_logits, sub_train_labels = subsample_logits_and_labels(train_logits, train_labels, n_samples, rs=rs)
                cal_test_logits = run_calibration(sub_train_logits, sub_train_labels, test_logits, test_labels, method=method)
                np.save(os.path.join(root_save_path,str(config["id"]),"calibration",f"test.{method}.{n_samples}.npy"),cal_test_logits)

                
def subsample_logits_and_labels(logits, labels, n_samples, rs):
    idx = rs.permutation(len(labels))[:n_samples]
    new_logits = logits[idx,:].copy()
    new_labels = labels[idx].copy()
    return new_logits, new_labels

def run_calibration(train_logits, train_labels, test_logits, test_labels, method):
    num_classes = train_logits.shape[1]
    train_logits = torch.from_numpy(train_logits)
    train_labels = torch.from_numpy(train_labels)
    test_logits = torch.from_numpy(test_logits)
    test_labels = torch.from_numpy(test_labels)

    if method == "affine_bias_only":
        model = AffineCalibrator(num_classes, model="bias only", psr="log-loss", maxiters=100, lr=1e-4, tolerance=1e-6)
        model.train_calibrator(train_logits, train_labels)
    elif method == "affine_vector_scaling":
        model = AffineCalibrator(num_classes, model="vector scaling", psr="log-loss", maxiters=100, lr=1e-4, tolerance=1e-6)
        model.train_calibrator(train_logits, train_labels)
    elif method in "UCPA":
        model = UCPACalibrator(num_classes, max_iters=10, tolerance=1e-6)
        model.train_calibrator(train_logits, priors=None)
    elif method in "SUCPA":
        model = UCPACalibrator(num_classes, max_iters=10, tolerance=1e-6)
        priors = torch.bincount(test_labels, minlength=num_classes)
        model.train_calibrator(train_logits, priors=priors)
    elif method in "UCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1, tolerance=1e-6)
        model.train_calibrator(train_logits, priors=None)
    elif method in "SUCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1, tolerance=1e-6)
        priors = torch.bincount(test_labels, minlength=num_classes)
        model.train_calibrator(train_logits, priors=priors)
    else:
        raise ValueError(f"Calibration method {method} not supported.")
    
    cal_test_logits = model.calibrate(test_logits)
    cal_test_logits = cal_test_logits.numpy()

    return cal_test_logits
        

if __name__ == "__main__":
    main()