
import argparse
from copy import deepcopy
import json
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from ucpa.calibration.affine.psrcal.losses import LogLoss, Brier
from sklearn.metrics import accuracy_score
import pandas as pd

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

    scores_records = []
    evaluation_metrics = ["cross-entropy", "accuracy"]
    for config in args.configs_dicts:

        # Results path for this config
        os.makedirs(os.path.join(root_save_path,str(config["id"]),"calibration"), exist_ok=True)

        # Load results and calibrate
        train_results = {
            "logits": np.load(os.path.join(root_save_path,str(config["id"]),f"train.logits.npy")),
            "labels": np.load(os.path.join(root_save_path,str(config["id"]),f"train.labels.npy"))
        }
        test_results = {
            "logits": np.load(os.path.join(root_save_path,str(config["id"]),f"test.logits.npy")),
            "labels": np.load(os.path.join(root_save_path,str(config["id"]),f"test.labels.npy"))
        }

        print("Running calibration...")
        results = run_calibration_with_bootstrap(train_results, test_results, bootstrap=100, num_train_samples=config["num_calibration_train_samples"], random_state=args.seed)
        with open(os.path.join(root_save_path,str(config["id"]),"calibration/cal_probs.pkl"), "wb") as f:
            pickle.dump(results,f)

        scores = compute_metrics(results, {"dataset": args.dataset, "model": args.model, "random_state": args.seed, "n_shots": config["n_shots"]}, evaluation_metrics)
        scores_records.extend(scores)

    # Save Scores
    scores_df = pd.DataFrame(scores_records)
    scores_df.to_csv(os.path.join(root_save_path,"calibration_scores.csv"), index=False)
    print("Done!")

def compute_metrics(results, config, evaluation_metrics):
    scores = []
    for i, result in enumerate(results):
        for key in result.keys():
            if key in ["boot_idx", "test_labels"]:
                continue

            this_prob_type = {
                **deepcopy(config),
                "prob_type": key,
                "bootstrap_iter": i
            }
            for metric in evaluation_metrics:
                if metric  == "cross-entropy":
                    metric_fn = lambda probs, labels: LogLoss(torch.from_numpy(np.log(probs)), torch.from_numpy(labels), norm=True, priors=None).item()
                elif metric == "brier":
                    metric_fn = lambda probs, labels: Brier(np.log(probs), labels, norm=True, priors=None)
                elif metric == "accuracy":
                    metric_fn = lambda probs, labels: accuracy_score(np.argmax(probs, axis=1), labels, normalize=True, sample_weight=None)
                else:
                    raise ValueError(f"Metric {metric} not supported!")
            
                score = metric_fn(result[key], result["test_labels"])
                this_prob_type[f"score:{metric}"] = float(score)

            scores.append(this_prob_type)

    return scores

def run_calibration_with_bootstrap(train_results, test_results, bootstrap, num_train_samples, random_state=None):
    test_probs = np.exp(test_results["logits"])
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    test_labels = test_results["labels"]
    
    train_probs = np.exp(train_results["logits"])
    train_probs = train_probs / train_probs.sum(axis=1, keepdims=True)
    train_labels = train_results["labels"]

    rs = np.random.RandomState(random_state)
    train_models_dict = {}
    for n in num_train_samples:
        train_idx = rs.choice(len(train_probs), n, replace=False)
        subsample_probs = train_probs[train_idx]
        subsample_labels = train_labels[train_idx]
        noscalecalmodel = train_calibrator_from_probs(subsample_probs, subsample_labels, calmethod="affine_bias_only")
        reestmodel = train_calibrator_from_probs(subsample_probs, calmethod="UCPA-naive")
        reestmodelwithpriors = train_calibrator_from_probs(subsample_probs, subsample_labels, calmethod="SUCPA-naive")
        reestiterative = train_calibrator_from_probs(subsample_probs, calmethod="UCPA")
        reestiterativewithpriors = train_calibrator_from_probs(subsample_probs, subsample_labels, calmethod="SUCPA")
        train_models_dict[n] = (noscalecalmodel, reestmodel, reestmodelwithpriors, reestiterative, reestiterativewithpriors)

    if bootstrap is not None:
        boots_idx = [rs.choice(len(test_labels), len(test_labels), replace=True) for _ in range(bootstrap)]
    else:
        boots_idx = [None]

    boots_results = []
    for bi in boots_idx:
        test_probs_boots = test_probs[bi].copy() if bi is not None else test_probs.copy()
        test_labels_boots = test_labels[bi].copy() if bi is not None else test_labels.copy()
        result = calibrate_reestimate_all(
            test_probs_boots, 
            test_labels_boots, 
            train_models_dict,
        )
        result["boot_idx"] = bi
        boots_results.append(result)
    
    return boots_results

def calibrate_reestimate_all(
    test_probs, 
    test_labels,
    train_models_dict, 
):
    
    test_probs_noscalecal_train = {}
    test_probs_reest_train = {}
    test_probs_reestwithpriors_train = {}
    test_probs_reestiterative_train = {}
    test_probs_reestiterativewithpriors_train = {}
    for n, (noscalecalmodel, reestmodel, reestmodelwithpriors, reestiterative, reestiterativewithpriors) in train_models_dict.items():
        test_probs_noscalecal_train[n] = calibrate_probs_from_trained_model(test_probs,noscalecalmodel)
        test_probs_reest_train[n] = calibrate_probs_from_trained_model(test_probs,reestmodel)
        test_probs_reestwithpriors_train[n] = calibrate_probs_from_trained_model(test_probs,reestmodelwithpriors)
        test_probs_reestiterative_train[n] = calibrate_probs_from_trained_model(test_probs,reestiterative)
        test_probs_reestiterativewithpriors_train[n] = calibrate_probs_from_trained_model(test_probs,reestiterativewithpriors)

    test_probs_original = test_probs.copy()

    return {
        "test_probs_original": test_probs_original,
        "test_labels": test_labels,
        **{f"test_probs_noscalecal_train_{n}": test_probs_noscalecal_train[n] for n in train_models_dict},
        **{f"test_probs_reest_train_{n}": test_probs_reest_train[n] for n in train_models_dict},
        **{f"test_probs_reestwithpriors_train_{n}": test_probs_reestwithpriors_train[n] for n in train_models_dict},
        **{f"test_probs_reestiterative_train_{n}": test_probs_reestiterative_train[n] for n in train_models_dict},
        **{f"test_probs_reestiterativewithpriors_train_{n}": test_probs_reestiterativewithpriors_train[n] for n in train_models_dict},
    }


def train_calibrator_from_probs(probs, labels=None, calmethod="affine_bias_only"):
    logprobs = torch.log(torch.from_numpy(probs))
    if labels is not None:
        labels = torch.from_numpy(labels)
    num_classes = logprobs.shape[1]
    if calmethod == "affine_bias_only":
        model = AffineCalibrator(num_classes, model="bias only", psr="log-loss")
        model.train_calibrator(logprobs, labels)
    elif calmethod == "affine_vector_scaling":
        model = AffineCalibrator(num_classes, model="vector scaling", psr="log-loss")
        model.train_calibrator(logprobs, labels)
    elif calmethod in "UCPA":
        model = UCPACalibrator(num_classes, max_iters=20, tolerance=1e-6)
        model.train_calibrator(logprobs, priors=None)
    elif calmethod in "SUCPA":
        model = UCPACalibrator(num_classes, max_iters=20, tolerance=1e-6)
        priors = torch.bincount(labels, minlength=num_classes)
        model.train_calibrator(logprobs, priors=priors)
    elif calmethod in "UCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1)
        model.train_calibrator(logprobs, priors=None)
    elif calmethod in "SUCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1)
        priors = torch.bincount(labels, minlength=num_classes)
        model.train_calibrator(logprobs, priors=priors)
    else:
        raise ValueError(f"Calibration method {calmethod} not supported.")
    return model


def calibrate_probs_from_trained_model(probs, model):
    logprobs = torch.log(torch.from_numpy(probs))
    cal_logprobs = model.calibrate(logprobs)
    cal_probs = torch.softmax(cal_logprobs, dim=1).numpy()
    return cal_probs
    

if __name__ == "__main__":
    main()