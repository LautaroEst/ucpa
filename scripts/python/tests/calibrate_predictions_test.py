

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append("../efficient-reestimation")
from src.utils import parse_calibration_args, get_results_ids_from_config
import pickle

from ucpa.calibration import AffineCalibrator, UCPACalibrator, ReestimatorIterative
from ucpa.calibration_old import train_calibrator_from_probs, train_reestimator_from_probs, train_reestimator_iter_from_probs, calibrate_probs_from_trained_model, reestimate_probs_from_trained_model

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default=".")
    parser.add_argument("--experiment_name", type=str, default="paper_results")
    parser.add_argument("--model", type=str, default="gpt2-xl")
    parser.add_argument("--dataset", type=str, default="tony_zhao_sst2")
    parser.add_argument("--config", type=str, default="configs/paper_results/gpt2-xl_tony_zhao_sst2.jsonl")
    parser.add_argument("--seed", type=int, default=82033)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs_dicts = [json.loads(config) for config in f.read().splitlines()]
    setattr(args,"configs_dicts",configs_dicts)
    
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()

    with open(os.path.join("../efficient-reestimation/configs/models/gpt2-xl_trec_sst2.json")) as experiment_file:
        experiment_config = json.load(experiment_file)
    results_ids = get_results_ids_from_config("../efficient-reestimation", experiment_config)

    for result_id in results_ids:

        with open(f"../efficient-reestimation/results/train_test/{result_id}/config.json", "r") as f:
            original_config = json.load(f)

        if original_config["model"] != args.model or f"tony_zhao_" + original_config["dataset"] != args.dataset:
            continue

        # Results path for this config
        root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.dataset, args.model, str(original_config["random_state"]))
        os.makedirs(os.path.join(root_save_path,f"{original_config['n_shots']}_shots","calibration"), exist_ok=True)

        # Random state
        rs = np.random.RandomState(original_config["random_state"])
        
        # Load results and calibrate
        with open(os.path.join(f"../efficient-reestimation/results/train_test/{result_id}/train.pkl"), "rb") as f:
            train_results = pickle.load(f)
            train_logits = np.log(train_results["train_probs"])
            train_labels = train_results["train_labels"]
            np.save(os.path.join(root_save_path,f"{original_config['n_shots']}_shots",f"train.logits.npy"),train_logits)
            np.save(os.path.join(root_save_path,f"{original_config['n_shots']}_shots",f"train.labels.npy"),train_labels)
        with open(os.path.join(f"../efficient-reestimation/results/train_test/{result_id}/test.pkl"), "rb") as f:
            test_results = pickle.load(f)
            test_logits = np.log(test_results["test_probs"])
            test_labels = test_results["test_labels"]
            np.save(os.path.join(root_save_path,f"{original_config['n_shots']}_shots",f"test.logits.npy"),test_logits)
            np.save(os.path.join(root_save_path,f"{original_config['n_shots']}_shots",f"test.labels.npy"),test_labels)

        with open(os.path.join(f"../efficient-reestimation/results/calibrated/logloss_noboots/{result_id}.pkl"), "rb") as f:
            original_results = pickle.load(f)

        tqdm_num_samples = tqdm(args.configs_dicts[0]["num_calibration_train_samples"])
        for n_samples in tqdm_num_samples:
            # sub_train_logits, sub_train_labels = subsample_logits_and_labels(train_logits, train_labels, n_samples, rs=rs)
            train_idx = original_results[0][f"train_idx_{n_samples}"]
            sub_train_logits, sub_train_labels = train_logits[train_idx,:].copy(), train_labels[train_idx].copy()
            for method in args.configs_dicts[0]["calibration_methods"]:
                tqdm_num_samples.set_description(desc=f"{method} ({n_samples} samples): ")
                cal_test_logits = run_calibration(sub_train_logits, sub_train_labels, test_logits, method=method)
                np.save(os.path.join(root_save_path,f"{original_config['n_shots']}_shots","calibration",f"test.{method}.{n_samples}.npy"),cal_test_logits)
                if (np.abs(original_results[0][method2name(method,n_samples)] - np.exp(cal_test_logits)) < 1e-4).sum() != np.prod(list(cal_test_logits.shape)):
                    import pdb; pdb.set_trace()
                
def method2name(method,num_samples):
    if method == "affine_bias_only":
        return f"test_probs_noscalecal_train_{num_samples}"
    elif method == "UCPA":
        return f"test_probs_reestiterative_train_{num_samples}"
    elif method == "UCPA-naive":
        return f"test_probs_reest_train_{num_samples}"
    elif method == "SUCPA":
        return f"test_probs_reestiterativewithpriors_train_{num_samples}"
    elif method == "SUCPA-naive":
        return f"test_probs_reestwithpriors_train_{num_samples}"
    else:
        raise ValueError(f"Method {method} not supported")


def subsample_logits_and_labels(logits, labels, n_samples, rs):
    idx = rs.choice(len(labels), n_samples, replace=False)
    new_logits = logits[idx,:].copy()
    new_labels = labels[idx].copy()
    return new_logits, new_labels

def run_calibration(train_logits, train_labels, test_logits, method):

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
    
    cal_probs = torch.softmax(cal_logits,dim=1).numpy()
    cal_logprobs = np.log(cal_probs)
    return cal_logprobs




if __name__ == "__main__":
    main()