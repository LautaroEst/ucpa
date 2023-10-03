

import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np

def norm_cross_entropy(probs, labels):
    logits = torch.log(torch.from_numpy(probs))
    labels = torch.from_numpy(labels)
    score = F.cross_entropy(logits, labels, reduction="mean")
    priors = torch.bincount(labels,minlength=logits.shape[1]) / logits.shape[0]
    dummy_score = - (priors * torch.log(priors)).sum()
    score = score / dummy_score
    return score.item()

def accuracy(probs, labels):
    logits = torch.log(torch.from_numpy(probs))
    labels = torch.from_numpy(labels)
    score = torch.mean((torch.max(logits,dim=1).indices == labels).type(torch.float))
    return score.item()

def main():
    # "dataset": "sst2", "model": "gpt2-xl", "n_shots": 8, "random_state": 95931
    original_results_id = "152475582312899686340520115511974223940"
    with open(f"../efficient-reestimation/results/calibrated/logloss_100boots/{original_results_id}.pkl", "rb") as f:
        original_results = pickle.load(f)
    with open(f"../efficient-reestimation/results/train_test/{original_results_id}/config.json", "r") as f:
        original_config = json.load(f)
    original_results_all = pd.read_csv("../efficient-reestimation/results/gpt2-xl_trec_sst2_logloss_100boots.csv")
    uncal_probs = original_results_all.loc[
        (original_results_all["dataset"] == original_config["dataset"]) & \
        (original_results_all["model"] == original_config["model"]) & \
        (original_results_all["n_shots"] == original_config["n_shots"]) & \
        (original_results_all["random_state"] == original_config["random_state"]) & \
        (original_results_all["prob_type"] == "test_probs_original"),["score:cross-entropy","score:accuracy"]
    ]

    results = {"cross_entropy": [], "accuracy": []}
    for i in range(len(original_results)):
        probs = original_results[i]["test_probs_original"]
        labels = original_results[i]["test_labels"]
        results["cross_entropy"].append(norm_cross_entropy(probs, labels))
        results["accuracy"].append(accuracy(probs, labels))
    results = pd.DataFrame(results)

    print(np.sum(np.abs(uncal_probs.values - results.values) > 1e-5))



if __name__ == "__main__":
    main()