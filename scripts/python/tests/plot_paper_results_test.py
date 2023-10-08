
import argparse
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from ucpa.calibration_old.expected_cost.psrcal_wrappers import LogLoss
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../efficient-reestimation")
from src.utils import parse_calibration_args, get_results_ids_from_config
from scipy.special import logsumexp

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default=".")
    parser.add_argument("--experiment_name", type=str, default="paper_results")
    parser.add_argument("--models", type=str, default="gpt2-xl")
    parser.add_argument("--datasets", type=str, default="tony_zhao_trec tony_zhao_sst2")
    parser.add_argument("--metrics", type=str, default="norm_cross_entropy error_rate")
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()

    return args


metric_short2name = {
    "norm_cross_entropy": "Normalized Cross-Entropy",
    "error_rate": "Error Rate",
    "accuracy": "Accuracy"
}

dataset_short2name = {
    "tony_zhao_trec": "TREC",
    "tony_zhao_sst2": "SST-2",
    "tony_zhao_agnews": "AGNews",
    "tony_zhao_dbpedia": "DBPedia"
}



def main():
    
    args = parse_args()
    models = args.models.split(" ")
    datasets = args.datasets.split(" ")
    metrics = args.metrics.split(" ")

    print("Collecting results...")
    if args.bootstrap == 0:
        results = collect_results(args.root_directory, args.experiment_name, models, datasets, metrics, bootstrap=False)
    elif args.bootstrap > 0:
        results = collect_results(args.root_directory, args.experiment_name, models, datasets, metrics, bootstrap=True, N_bootstrap=args.bootstrap)
    else:
        raise ValueError("Bootstrap number must be positive or 0 (no bootstrap)")

    for model in models:
        # Number of samples vs metric
        print("Plotting samples vs metrics...")
        fig = plot_num_samples_vs_metrics(args.root_directory, args.experiment_name, results, model, datasets, metrics)
        # Number of shots vs metric
        print("Plotting shots vs metrics...")
        fig = plot_num_shots_vs_metrics(args.root_directory, args.experiment_name, results, model, datasets, metrics)
        

def plot_num_samples_vs_metrics(root_directory, experiment_name, results, model, datasets, metrics):

    method2format = {
        "no_adaptation": {"color": "black", "linestyle": "-", "label": "No Adaptation", "linewidth": 2},
        "affine_bias_only": {"color": "tab:orange", "linestyle": "-", "label": "Calibration with $\\alpha=1$"},
        "UCPA-naive": {"color": "tab:blue", "linestyle": "-", "label": "UCPA naive", "linewidth": 2},
        "SUCPA-naive": {"color": "tab:blue", "linestyle": "--", "label": "SUCPA naive", "linewidth": 2, "marker": ".", "markersize": 10},
        "UCPA": {"color": "tab:red", "linestyle": "-", "label": "UCPA", "linewidth": 2},
        "SUCPA": {"color": "tab:red", "linestyle": "--", "label": "SUCPA", "linewidth": 2, "marker": ".", "markersize": 10},
    }
    num_shots = results.index.get_level_values("num_shots").unique()
    num_samples = results.index.get_level_values("num_samples").unique().values
    num_samples = np.array([s for s in num_samples if s != -1])
    for n in num_shots:
        fig, ax = plt.subplots(len(metrics),len(datasets),figsize=(20, 10),sharex=True,sharey=False)
        ax = ax.reshape(len(metrics),len(datasets))
        for j, dataset in enumerate(datasets):
            for i, metric in enumerate(metrics):
                methods = results.index.get_level_values("method").unique()
                for method in methods:
                    if method == "no_adaptation":
                        mean = results.loc[(model,dataset,method,n,-1),(f"metric:{metric}","mean")]
                        means = np.array([mean for _ in num_samples])
                        std = results.loc[(model,dataset,method,n,-1),(f"metric:{metric}","std")]
                        stds = np.array([std for _ in num_samples])
                    else:
                        means = results.loc[(model,dataset,method,n,slice(None)),(f"metric:{metric}","mean")].values.reshape(-1)
                        stds = results.loc[(model,dataset,method,n,slice(None)),(f"metric:{metric}","std")].values.reshape(-1)

                    ax[i,j].plot(num_samples, means, **method2format[method])
                    ax[i,j].fill_between(num_samples, means-stds, means+stds, alpha=0.2, color=method2format[method]["color"])
                ax[i,j].grid(True)
                ax[i,j].set_xscale("log")
                ax[i,j].set_xticks(num_samples)
                ax[i,j].set_xticklabels(num_samples, fontsize=12, rotation=45)
                ax[i,j].set_xlim(num_samples[0],num_samples[-1])
                if i == 0:
                    ax[i,j].set_title(dataset_short2name[dataset], fontsize=18)
            if j == 0:
                ax[i,j].set_ylabel(metric_short2name[metric], fontsize=18)

        handles, labels = ax[i,j].get_legend_handles_labels()
        first_handle = handles.pop(0)
        handles.append(first_handle)
        first_label = labels.pop(0)
        labels.append(first_label)
        fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
        fig.supxlabel("Number of samples", fontsize=20)
        fig.savefig(os.path.join(root_directory,"results_test",experiment_name,f"{model}_{n}-shots_samples_vs_metric.png"))


def plot_num_shots_vs_metrics(root_directory, experiment_name, results, model, datasets, metrics):

    method2format = {
        "no_adaptation": {"color": "black", "linestyle": "-", "label": "No Adaptation", "linewidth": 2},
        "affine_bias_only": {"color": "tab:green", "linestyle": "-", "label": "Calibration with $\\alpha=1$"},
        "UCPA-naive": {"color": "tab:blue", "linestyle": "-", "label": "UCPA naive", "linewidth": 2},
        "SUCPA-naive": {"color": "tab:blue", "linestyle": "--", "label": "SUCPA naive", "linewidth": 2, "marker": ".", "markersize": 10},
        "UCPA": {"color": "tab:red", "linestyle": "-", "label": "UCPA", "linewidth": 2},
        "SUCPA": {"color": "tab:red", "linestyle": "--", "label": "SUCPA", "linewidth": 2, "marker": ".", "markersize": 10},
    }

    num_samples = results.index.get_level_values("num_samples").unique()
    num_samples = sorted([s for s in num_samples if s != -1])
    for n in num_samples:
        fig, ax = plt.subplots(len(metrics),len(datasets),figsize=(20, 10),sharex=True,sharey=False)
        ax = ax.reshape(len(metrics),len(datasets))
        for j, dataset in enumerate(datasets):
            for i, metric in enumerate(metrics):
                methods = results.index.get_level_values("method").unique()
                for method in methods:
                    if method == "no_adaptation":
                        num_shots = results.loc[(model,dataset,method,slice(None),-1),(f"metric:{metric}","mean")].index.get_level_values("num_shots").values
                        means = results.loc[(model,dataset,method,slice(None),-1),(f"metric:{metric}","mean")].values.reshape(-1)
                        stds = results.loc[(model,dataset,method,slice(None),-1),(f"metric:{metric}","std")].values.reshape(-1)
                    else:
                        num_shots = results.loc[(model,dataset,method,slice(None),n),(f"metric:{metric}","mean")].index.get_level_values("num_shots").values
                        means = results.loc[(model,dataset,method,slice(None),n),(f"metric:{metric}","mean")].values.reshape(-1)
                        stds = results.loc[(model,dataset,method,slice(None),n),(f"metric:{metric}","std")].values.reshape(-1)
                    
                    ax[i,j].plot(num_shots, means, **method2format[method])
                    ax[i,j].fill_between(num_shots, means-stds, means+stds, alpha=0.2, color=method2format[method]["color"])
                ax[i,j].grid(True)
                ax[i,j].set_xticks(num_shots)
                ax[i,j].set_xticklabels(num_shots, fontsize=12)
                ax[i,j].set_xlim(num_shots[0],num_shots[-1])
                if i == 0:
                    ax[i,j].set_title(dataset_short2name[dataset], fontsize=18)
                if j == 0:
                    ax[i,j].set_ylabel(metric_short2name[metric], fontsize=18)

        handles, labels = ax[i,j].get_legend_handles_labels()
        first_handle = handles.pop(0)
        handles.append(first_handle)
        first_label = labels.pop(0)
        labels.append(first_label)
        fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
        fig.supxlabel("Number of shots", fontsize=20)
        fig.savefig(os.path.join(root_directory,"results_test",experiment_name,f"{model}_{n}-samples_shots_vs_metric.png"))


def collect_results(root_directory, experiment_name, models, datasets, metrics, bootstrap=True, N_bootstrap=None):
    results = []
    for model in models:
        for dataset in datasets:
            results_dir = os.path.join(root_directory, "results_test", experiment_name, dataset, model)
            for seed in os.listdir(os.path.join(root_directory, "results_test", experiment_name, dataset, model)):
                rs = np.random.RandomState(int(seed))
                for n_shot in os.listdir(os.path.join(results_dir,seed)):
                    labels = np.load(os.path.join(results_dir,seed,n_shot,"test.labels.npy"))
                    boots_idx = [rs.choice(len(labels), len(labels), replace=True) for _ in range(N_bootstrap)] if bootstrap else [None]
                    results_filenames = os.listdir(os.path.join(results_dir,seed,n_shot,"calibration"))
                    results_filenames.append("no_adaptation")
                    for result in results_filenames:
                        if result == "no_adaptation":
                            method, num_samples = "no_adaptation", "-1"
                            result_logits = np.load(os.path.join(results_dir,seed,n_shot,"test.logits.npy"))
                        else:
                            method, num_samples = result.split(".")[1:3]
                            result_logits = np.load(os.path.join(results_dir,seed,n_shot,"calibration",result))
                        
                        for i, bi in enumerate(boots_idx):
                            results_dict = {f"metric:{metric}": compute_metric(result_logits, labels, metric, bootstrap_idx=bi) for metric in metrics}
                            results_dict["model"] = model
                            results_dict["dataset"] = dataset
                            results_dict["method"] = method
                            results_dict["num_shots"] = int(n_shot.split("_")[0])
                            results_dict["num_samples"] = int(num_samples)
                            results_dict["random_state"] = seed
                            results_dict["bootstrap_iter"] = i
                            results.append(results_dict)

    results = pd.DataFrame.from_records(results)
    results.to_csv(os.path.join(root_directory,"results_test",experiment_name,"results.csv"),index=False)
    results = results.groupby(by=["model","dataset","method","num_shots","num_samples"]).agg({f"metric:{metric}": ["mean","std"] for metric in metrics})
    return results

# def compute_metric(logits, labels, metric="cross_entropy", bootstrap_idx=None):
    
#     # if method == "no_adaptation":
#     #     with open(os.path.join("../efficient-reestimation/configs/models/gpt2-xl_trec_sst2.json")) as experiment_file:
#     #         experiment_config = json.load(experiment_file)
#     #     results_ids = get_results_ids_from_config("../efficient-reestimation", experiment_config)
#     #     for result_id in results_ids:
#     #         with open(f"../efficient-reestimation/results/train_test/{result_id}/config.json", "r") as f:
#     #             original_config = json.load(f)
#     #         if original_config["model"] == model \
#     #         and f"tony_zhao_" + original_config["dataset"] == dataset \
#     #         and original_config["random_state"] == int(seed) \
#     #         and original_config["n_shots"] == int(n_shot.split("_")[0]):
#     #             break
#     #     with open(os.path.join(f"../efficient-reestimation/results/calibrated/logloss_noboots/{result_id}.pkl"), "rb") as f:
#     #         cal_results = pickle.load(f)
#     #     test_probs = cal_results[0]["test_probs_original"]
#     #     import pdb; pdb.set_trace()

#     if bootstrap_idx is not None:
#         logits = logits[bootstrap_idx,:]
#         labels = labels[bootstrap_idx]

#     if metric == "cross_entropy":
#         logprobs = np.exp(logits)
#         logprobs = np.log(logprobs / np.sum(logprobs,axis=1,keepdims=True))
#         score = LogLoss(logprobs, labels, norm=False, priors=None)
#     elif metric == "norm_cross_entropy":
#         logprobs = np.exp(logits)
#         logprobs = np.log(logprobs / np.sum(logprobs,axis=1,keepdims=True))
#         score = LogLoss(logprobs, labels, norm=True, priors=None)
#     elif metric == "accuracy":
#         score = accuracy_score(np.argmax(logits, axis=1), labels, normalize=True, sample_weight=None)
#     elif metric == "error_rate":
#         score = accuracy_score(np.argmax(logits, axis=1), labels, normalize=True, sample_weight=None)
#         score = 1 - score
#     else:
#         raise ValueError(f"Metric {metric} not supported.")
    
#     score = score.item()
#     return score



def compute_metric(logits, labels, metric="cross_entropy", bootstrap_idx=None):
    
    if bootstrap_idx is not None:
        logits = logits[bootstrap_idx,:]
        labels = labels[bootstrap_idx]

    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)

    if metric == "cross_entropy":
        score = F.cross_entropy(logits, labels, reduction="mean")
    elif metric == "norm_cross_entropy":
        score = F.cross_entropy(logits, labels, reduction="mean")
        priors = torch.bincount(labels,minlength=logits.shape[1]) / logits.shape[0]
        dummy_score = - (priors * torch.log(priors)).sum()
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