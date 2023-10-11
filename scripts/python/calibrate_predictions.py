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
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    with open(os.path.join(args.root_directory,f"configs/{args.experiment_name}/{args.model}.json"), "r") as f:
        configs_dicts = json.load(f)
    setattr(args,"configs_dicts",configs_dicts)
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(" ")]

    for seed in seeds:
        
        # Root path to save results:
        root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.model, str(seed))
        
        for config in args.configs_dicts:

            # Results path for this config
            os.makedirs(os.path.join(root_save_path, config["id"]), exist_ok=True)
            
            # Random state
            rs = np.random.RandomState(seed+config["random_state"])

            # Load logits and labels
            train_logits = np.load(os.path.join(config["train"].format(root_directory=args.root_directory,seed=seed),"logits.npy"))
            train_labels = np.load(os.path.join(config["train"].format(root_directory=args.root_directory,seed=seed),"labels.npy"))
            test_logits = np.load(os.path.join(config["test"].format(root_directory=args.root_directory,seed=seed),"logits.npy"))
            test_labels = np.load(os.path.join(config["test"].format(root_directory=args.root_directory,seed=seed),"labels.npy"))
            
            # Save them in the new results directory
            np.save(os.path.join(root_save_path,str(config["id"]),f"train.logits.original.npy"),train_logits)
            np.save(os.path.join(root_save_path,str(config["id"]),f"train.labels.original.npy"),train_labels)
            np.save(os.path.join(root_save_path,str(config["id"]),f"test.logits.original.npy"),test_logits)
            np.save(os.path.join(root_save_path,str(config["id"]),f"test.labels.original.npy"),test_labels)

            # Train samples:
            train_samples = tqdm(config["num_train_samples"])
            for n_samples in train_samples:
                train_idx = rs.choice(len(train_labels), n_samples, replace=False)
                sub_train_logits = train_logits[train_idx,:].copy()
                sub_train_labels = train_labels[train_idx].copy()
                for method in config["methods"]:
                    train_samples.set_description(desc=f"{method} ({n_samples} samples)")
                    cal_test_logits = run_calibration(sub_train_logits, sub_train_labels, test_logits, method=method)
                    np.save(os.path.join(root_save_path,str(config["id"]),f"test.{method}.{n_samples}.npy"),cal_test_logits)
                np.save(os.path.join(root_save_path,str(config["id"]),"train.idx.original.npy"),train_idx)
                train_samples.set_description(desc=config["id"])
            
            # Save config
            with open(os.path.join(root_save_path, config["id"], "config.json"), "w") as f:
                json.dump(config,f,indent=4)


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