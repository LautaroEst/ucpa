

import argparse
import os
import numpy as np
from ucpa.calibration import AffineCalibrator, UCPACalibrator
import torch
import torch.nn.functional as F
import pandas as pd

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    args = parser.parse_args()
    return args


def main():
    model = "meta-llama--Llama-2-7b-hf"
    dataset = "tony_zhao_sst2"
    p0_values = [0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1]
    n_shots = 8
    N_train = 600
    N_test = 1000
    seeds = [82033, 12782, 1263, 987, 12299]

    args = parse_args()
    test_predictions = os.path.join(args.root_directory,"results/predict_with_model",model,"{seed}",f"{dataset}_test_{n_shots}_shot","{prediction}.npy")
    train_predictions = os.path.join(args.root_directory,"results/predict_with_model",model,"{seed}",f"{dataset}_train_queries_with_prompt_{n_shots}_shot","{prediction}.npy")

    results = {"p0": [], "p0_estimate": [], "seed": [], "ce": [], "er": [], "method": []}
    for seed in seeds:
        rs = np.random.RandomState(seed)
        for p0 in p0_values:
            test_logits = np.load(test_predictions.format(seed=seed,prediction="logits"))
            test_labels = np.load(test_predictions.format(seed=seed,prediction="labels"))
            test_logits, test_labels = create_dataset_with_proportion(test_logits, test_labels, p0, N_test, rs=rs)
            
            train_logits = np.load(train_predictions.format(seed=seed,prediction="logits"))
            train_labels = np.load(train_predictions.format(seed=seed,prediction="labels"))
            train_logits, train_labels = create_dataset_with_proportion(train_logits, train_labels, p0, N_train, rs=rs)

            model = UCPACalibrator(num_classes=2, max_iters=20, tolerance=1e-6)
            train_logits = torch.from_numpy(train_logits)
            test_logits = torch.from_numpy(test_logits)
            test_labels = torch.from_numpy(test_labels)
            model.train_calibrator(train_logits)
            cal_test_logits = model.calibrate(test_logits)
            
            results["seed"].append(seed)
            results["p0"].append(p0)
            results["p0_estimate"].append(torch.bincount(test_labels,minlength=2)[0].item() / len(test_labels))
            results["method"].append("no_adaptation")
            results["ce"].append(compute_metric(test_logits, test_labels, metric="norm_cross_entropy"))
            results["er"].append(compute_metric(test_logits, test_labels, metric="error_rate"))

            results["seed"].append(seed)
            results["p0"].append(p0)
            results["p0_estimate"].append(torch.bincount(test_labels,minlength=2)[0].item() / len(test_labels))
            results["method"].append("UCPA")
            results["ce"].append(compute_metric(cal_test_logits, test_labels, metric="norm_cross_entropy"))
            results["er"].append(compute_metric(cal_test_logits, test_labels, metric="error_rate"))

    results = pd.DataFrame(results)
    import pdb; pdb.set_trace()
    print(results)

            





def create_dataset_with_proportion(logits, labels, p, N, rs):
    idx = np.arange(len(labels))
    p0_index = rs.permutation(idx[labels == 0])[:max([1,int(np.ceil(p*N))])]
    p1_index = rs.permutation(idx[labels == 1])[:max([1,int(np.ceil((1-p)*N))])]
    if len(p0_index) > len(p1_index):
        idx = np.hstack([p1_index,p0_index])[:N]
    else:
        idx = np.hstack([p0_index,p1_index])[:N]
    labels = labels[idx]
    logits = logits[idx,:]
    return logits, labels

def compute_metric(logits, labels, metric="cross_entropy"):
    
    if metric == "cross_entropy":
        score = F.cross_entropy(logits, labels, reduction="mean")
    elif metric == "norm_cross_entropy":
        score = F.cross_entropy(logits, labels, reduction="mean")
        priors = torch.bincount(labels,minlength=logits.shape[1]) / logits.shape[0]
        dummy_score = F.cross_entropy(priors.repeat(logits.shape[0],1), labels)
        score = score / dummy_score
    elif metric == "accuracy":
        score = torch.mean((torch.max(logits,dim=1).indices == labels).type(torch.float))
    elif metric == "error_rate":
        score = torch.mean((torch.max(logits,dim=1).indices == labels).type(torch.float))
        score = 1 - score
    else:
        raise ValueError(f"Metric {metric} not supported.")
    
    score = score.item()
    return score



if __name__ == "__main__":
    main()