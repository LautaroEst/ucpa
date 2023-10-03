
import argparse
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

    print("Collecting results...")
    if args.bootstrap == 0:
        results = collect_results(args.root_directory, args.experiment_name, models, datasets, metrics, seeds, bootstrap=False)
    elif args.bootstrap > 0:
        results = collect_results(args.root_directory, args.experiment_name, models, datasets, metrics, seeds, bootstrap=True, N_bootstrap=args.bootstrap)
    else:
        raise ValueError("Bootstrap number must be positive or 0 (no bootstrap)")
    results.reset_index(drop=False,inplace=False).to_csv(os.path.join(args.root_directory,"results",args.experiment_name,"results.csv"))

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
                    ax[i,j].set_title(dataset, fontsize=18)
            if j == 0:
                ax[i,j].set_ylabel(metric, fontsize=18)

        handles, labels = ax[i,j].get_legend_handles_labels()
        first_handle = handles.pop(0)
        handles.append(first_handle)
        first_label = labels.pop(0)
        labels.append(first_label)
        fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
        fig.supxlabel("Number of samples", fontsize=20)
        fig.savefig(os.path.join(root_directory,"results",experiment_name,f"{model}_{n}-shots_samples_vs_metric.png"))


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
                    ax[i,j].set_title(dataset, fontsize=18)
            if j == 0:
                ax[i,j].set_ylabel(metric, fontsize=18)

        handles, labels = ax[i,j].get_legend_handles_labels()
        first_handle = handles.pop(0)
        handles.append(first_handle)
        first_label = labels.pop(0)
        labels.append(first_label)
        fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
        fig.supxlabel("Number of shots", fontsize=20)
        fig.savefig(os.path.join(root_directory,"results",experiment_name,f"{model}_{n}-samples_shots_vs_metric.png"))


def collect_results(root_directory, experiment_name, models, datasets, metrics, seeds, bootstrap=True, N_bootstrap=None):
    results = []
    for model in models:
        for dataset in datasets:
            results_dir = os.path.join(root_directory, "results", experiment_name, dataset, model)
            for seed in seeds:
                rs = np.random.RandomState(int(seed))
                for n_shot in os.listdir(os.path.join(results_dir,seed)):
                    labels = np.load(os.path.join(results_dir,seed,n_shot,"test.labels.npy"))
                    original_logits = np.load(os.path.join(results_dir,seed,n_shot,"test.logits.npy"))

                    if bootstrap:
                        for _ in range(N_bootstrap):
                            results_dict = {f"metric:{metric}": compute_metric(original_logits, labels, metric, bootstrap=True, rs=rs) for metric in metrics}
                            results_dict["model"] = model
                            results_dict["dataset"] = dataset
                            results_dict["num_shots"] = int(n_shot.split("_")[0])
                            results_dict["method"] = "no_adaptation"
                            results_dict["num_samples"] = -1
                            results.append(results_dict)
                    else:
                        results_dict = {f"metric:{metric}": compute_metric(original_logits, labels, metric, bootstrap=False, rs=rs) for metric in metrics}
                        results_dict["model"] = model
                        results_dict["dataset"] = dataset
                        results_dict["method"] = "no_adaptation"
                        results_dict["num_shots"] = int(n_shot.split("_")[0])
                        results_dict["num_samples"] = -1
                        results.append(results_dict)
                    
                    for result in os.listdir(os.path.join(results_dir,seed,n_shot,"calibration")):
                        method, num_samples = result.split(".")[1:3]
                        result_logits = np.load(os.path.join(results_dir,seed,n_shot,"calibration",result))
                        
                        if bootstrap:
                            for _ in range(N_bootstrap):
                                results_dict = {f"metric:{metric}": compute_metric(result_logits, labels, metric, bootstrap=True, rs=rs) for metric in metrics}
                                results_dict["model"] = model
                                results_dict["dataset"] = dataset
                                results_dict["method"] = method
                                results_dict["num_shots"] = int(n_shot.split("_")[0])
                                results_dict["num_samples"] = int(num_samples)
                                results.append(results_dict)
                        else:
                            results_dict = {f"metric:{metric}": compute_metric(result_logits, labels, metric, bootstrap=False, rs=rs) for metric in metrics}
                            results_dict["model"] = model
                            results_dict["dataset"] = dataset
                            results_dict["method"] = method
                            results_dict["num_shots"] = int(n_shot.split("_")[0])
                            results_dict["num_samples"] = int(num_samples)
                            results.append(results_dict)
    
    results = pd.DataFrame.from_records(results)
    results = results.groupby(by=["model","dataset","method","num_shots","num_samples"]).agg({f"metric:{metric}": ["mean","std"] for metric in metrics})
    return results


def compute_metric(logits, labels, metric="cross_entropy", bootstrap=True, rs=None):
    
    if bootstrap:
        if rs is None:
            boots_idx = np.random.choice(len(labels),len(labels),replace=True)
        else:
            boots_idx = rs.choice(len(labels),len(labels),replace=True)
        logits = logits[boots_idx,:]
        labels = labels[boots_idx]

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