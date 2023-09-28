
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--metrics", type=str, default="")
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    
    args = parse_args()
    models = args.models.split(" ")
    datasets = args.datasets.split(" ")
    metrics = args.metrics.split(" ")
    seeds = args.seeds.split(" ")

    results = []
    for model in models:
        for dataset in datasets:
            results_dir = os.path.join(args.root_directory, "results", args.experiment_name, dataset, args.models)
            for seed in seeds:
                rs = np.random.RandomState(int(seed))
                for n_shot in os.listdir(os.path.join(results_dir,seed)):
                    labels = np.load(os.path.join(results_dir,seed,n_shot,"test.labels.npy"))
                    original_logits = np.load(os.path.join(results_dir,seed,n_shot,"test.logits.npy"))

                    if args.bootstrap == 0:
                        results_dict = {f"metric:{metric}": compute_metric(original_logits, labels, metric, bootstrap=False, rs=rs) for metric in metrics}
                        results_dict["model"] = model
                        results_dict["dataset"] = dataset
                        results_dict["seed"] = int(seed)
                        results_dict["num_shots"] = int(n_shot.split("_")[0])
                        results_dict["method"] = "no_adaptation"
                        results_dict["num_samples"] = None
                        results.append(results_dict)
                    else:
                        for n in range(args.bootstrap):
                            results_dict = {f"metric:{metric}": compute_metric(original_logits, labels, metric, bootstrap=True, rs=rs) for metric in metrics}
                            results_dict["model"] = model
                            results_dict["dataset"] = dataset
                            results_dict["seed"] = int(seed)
                            results_dict["num_shots"] = int(n_shot.split("_")[0])
                            results_dict["method"] = "no_adaptation"
                            results_dict["num_samples"] = None
                            results.append(results_dict)
                    
                    for result in os.listdir(os.path.join(results_dir,seed,n_shot,"calibration")):
                        method, num_samples = result.split(".")[1:3]
                        result_logits = np.load(os.path.join(results_dir,seed,n_shot,"calibration",result))
                        
                        if args.bootstrap == 0:
                            results_dict = {f"metric:{metric}": compute_metric(result_logits, labels, metric, bootstrap=False, rs=rs) for metric in metrics}
                            results_dict["model"] = model
                            results_dict["dataset"] = dataset
                            results_dict["seed"] = int(seed)
                            results_dict["num_shots"] = int(n_shot.split("_")[0])
                            results_dict["method"] = method
                            results_dict["num_samples"] = int(num_samples)
                            results.append(results_dict)
                        else:
                            for n in range(args.bootstrap):
                                results_dict = {f"metric:{metric}": compute_metric(result_logits, labels, metric, bootstrap=True, rs=rs) for metric in metrics}
                                results_dict["model"] = model
                                results_dict["dataset"] = dataset
                                results_dict["seed"] = int(seed)
                                results_dict["num_shots"] = int(n_shot.split("_")[0])
                                results_dict["method"] = method
                                results_dict["num_samples"] = int(num_samples)
                                results.append(results_dict)
    results = pd.DataFrame.from_records(results)   
    results = results.groupby(by=["model","dataset","method","num_shots","num_samples"]).agg({f"metric:{metric}": ["mean","std"] for metric in metrics})
    
    for model in models:

        # Number of samples vs metric
        num_shots = results.index.get_level_values("num_shots").unique()
        for n in num_shots:
            fig, ax = plt.subplots(len(metrics),len(datasets),figsize=(20, 10),sharex=True,sharey=False)
            ax = ax.reshape(len(metrics),len(datasets))
            for j, dataset in enumerate(datasets):
                for i, metric in enumerate(metrics):
                    methods = results.index.get_level_values("method").unique()
                    for method in methods:
                        num_samples = results.loc[(model,dataset,method,n,slice(None)),(f"metric:{metric}","mean")].index.get_level_values("num_samples").values
                        means = results.loc[(model,dataset,method,n,slice(None)),(f"metric:{metric}","mean")].values.reshape(-1)
                        stds = results.loc[(model,dataset,method,n,slice(None)),(f"metric:{metric}","std")].values.reshape(-1)
                        ax[i,j].plot(num_samples, means, label=method)
                        ax[i,j].fill_between(num_samples, means-stds, means+stds, alpha=0.2)
                    ax[i,j].grid(True)
                    ax[i,j].set_ylabel(metric, fontsize=18)
                    if i == 0:
                        ax[i,j].set_title(dataset, fontsize=18)

            handles, labels = ax[i,j].get_legend_handles_labels()
            first_handle = handles.pop(0)
            handles.append(first_handle)
            first_label = labels.pop(0)
            labels.append(first_label)
            fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
            fig.supxlabel("Number of samples", fontsize=20)
            fig.savefig(os.path.join(args.root_directory,"results",args.experiment_name,f"{model}_{n}-shot_samples_vs_metric.png"))
        
        # Number of shots vs metric
        num_samples = results.index.get_level_values("num_samples").unique()
        for n in num_samples:
            fig, ax = plt.subplots(len(metrics),len(datasets),figsize=(20, 10),sharex=True,sharey=False)
            ax = ax.reshape(len(metrics),len(datasets))
            for j, dataset in enumerate(datasets):
                for i, metric in enumerate(metrics):
                    methods = results.index.get_level_values("method").unique()
                    for method in methods:
                        num_shots = results.loc[(model,dataset,method,slice(None),n),(f"metric:{metric}","mean")].index.get_level_values("num_shots").values
                        means = results.loc[(model,dataset,method,slice(None),n),(f"metric:{metric}","mean")].values.reshape(-1)
                        stds = results.loc[(model,dataset,method,slice(None),n),(f"metric:{metric}","std")].values.reshape(-1)
                        ax[i,j].plot(num_shots, means, label=method)
                        ax[i,j].fill_between(num_shots, means-stds, means+stds, alpha=0.2)
                    ax[i,j].grid(True)
                    ax[i,j].set_ylabel(metric, fontsize=18)
                    if i == 0:
                        ax[i,j].set_title(dataset, fontsize=18)

            handles, labels = ax[i,j].get_legend_handles_labels()
            first_handle = handles.pop(0)
            handles.append(first_handle)
            first_label = labels.pop(0)
            labels.append(first_label)
            fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
            fig.supxlabel("Number of shots", fontsize=20)
            fig.savefig(os.path.join(args.root_directory,"results",args.experiment_name,f"{model}_{n}-samples_shots_vs_metric.png"))






def compute_metric(logits, labels, metric="cross_entropy", bootstrap=True, rs=None):
    
    if bootstrap:
        df_boots = pd.DataFrame({"logits": logits, "labels": labels}).sample(n=logits.shape[0],replace=True,random_state=rs)
        logits, labels = df_boots["logits"].values, df_boots["labels"].values

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