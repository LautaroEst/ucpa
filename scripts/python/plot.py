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
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    with open(os.path.join(args.root_directory,args.config), "r") as f:
        configs_dicts = json.load(f)
    setattr(args,"configs_dicts",configs_dicts)
    return args


def main():
    
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(" ")]

    for config in args.configs_dicts:

        # Results directory:
        results_dir = os.path.join(args.root_directory, "results", args.experiment_name, config["id"])
        os.makedirs(results_dir, exist_ok=True)

        plot_type = config.pop("plot")
        if plot_type == "metrics_vs_samples":
            plot_metrics_vs_samples(args.root_directory,seeds=seeds,**config)
        elif plot_type == "metrics_vs_shots":
            plot_metrics_vs_shots(args.root_directory,seeds=seeds,**config)
        else:
            raise ValueError("Plot configuration not supported")


def plot_metrics_vs_samples(
    root_directory,
    id,
    logits_path,
    labels_path,
    models,
    datasets,
    num_samples,
    num_shots,
    methods,
    metrics,
    seeds,
    bootstrap=None,
    random_state=None
):
    rs = np.random.RandomState(random_state)
    for model in models:
        for n_shots in num_shots:
            fig, axes = plt.subplots(len(metrics),len(datasets),figsize=(20,5))
            if len(metrics) == 1 and len(datasets) == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(len(metrics),len(datasets))
            for d, dataset in enumerate(datasets):
                N = len(np.load(labels_path.format(
                    root_directory=root_directory,
                    model=models[0],
                    seed=seeds[0],
                    dataset=dataset,
                    num_shots=num_shots[0]
                )))
                boots_idx = [rs.choice(N, N, replace=True) for _ in range(bootstrap)] if bootstrap is not None else [None]
                for method in methods:
                    results = {
                        "num_samples": [], "seed": [], "bootstrap_index": [],
                        **{metric: [] for metric in metrics}
                    }
                    for seed in seeds:
                        labels = np.load(labels_path.format(
                            root_directory=root_directory,
                            model=model,
                            seed=seed,
                            dataset=dataset,
                            num_shots=n_shots
                        ))
                        for n_samples in num_samples:
                            if method == "no_adaptation":
                                logits = np.load(logits_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots,
                                    method="logits",
                                    num_samples="original"
                                ))
                            else:
                                logits = np.load(logits_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots,
                                    method=method,
                                    num_samples=n_samples
                                ))
                            for i, bi in enumerate(boots_idx):
                                results["num_samples"].append(n_samples)
                                results["seed"].append(seed)
                                results["bootstrap_index"].append(i)
                                for metric in metrics:
                                    score = compute_metric(logits, labels, metric, bootstrap_idx=bi)
                                    results[metric].append(score)
                    df = pd.DataFrame(results)
                    for m, metric in enumerate(metrics):
                        metric_vs_n_samples = df.groupby("num_samples").agg({metric: ["mean", "std"]})
                        means = metric_vs_n_samples.loc[num_samples,(metric,"mean")].values
                        stds = metric_vs_n_samples.loc[num_samples,(metric,"std")].values
                        axes[m,d].plot(num_samples,means,**methods[method])
                        axes[m,d].fill_between(num_samples, means-stds, means+stds, alpha=0.2,color=methods[method]["color"])
                        axes[m,d].grid(True)
                        axes[m,d].set_xscale("log")
                        axes[m,d].set_xticks(num_samples)
                        axes[m,d].set_xticklabels(num_samples, fontsize=12)
                        axes[m,d].set_xlim(num_samples[0],num_samples[-1])
                        if m == 0:
                            axes[m,d].set_title(datasets[dataset], fontsize=18)
                        if d == 0:
                            axes[m,d].set_ylabel(metrics[metric], fontsize=18)
            handles, labels = axes[m,d].get_legend_handles_labels()
            first_handle = handles.pop(0)
            handles.append(first_handle)
            first_label = labels.pop(0)
            labels.append(first_label)
            fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
            fig.supxlabel("Number of samples", fontsize=20)
            fig.savefig(f"results/plot/{id}-model_{model}-n_shots_{n_shots}")


def plot_metrics_vs_shots(
    root_directory,
    id,
    logits_path,
    labels_path,
    models,
    datasets,
    num_samples,
    num_shots,
    methods,
    metrics,
    seeds,
    bootstrap=None,
    random_state=None
):
    rs = np.random.RandomState(random_state)
    for model in models:
        for n_samples in num_samples:
            fig, axes = plt.subplots(len(metrics),len(datasets),figsize=(20,5))
            if len(metrics) == 1 and len(datasets) == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(len(metrics),len(datasets))
            for d, dataset in enumerate(datasets):
                N = len(np.load(labels_path.format(
                    root_directory=root_directory,
                    model=models[0],
                    seed=seeds[0],
                    dataset=dataset,
                    num_shots=num_shots[0]
                )))
                boots_idx = [rs.choice(N, N, replace=True) for _ in range(bootstrap)] if bootstrap is not None else [None]
                for method in methods:
                    results = {
                        "num_shots": [], "seed": [], "bootstrap_index": [],
                        **{metric: [] for metric in metrics}
                    }
                    for seed in seeds:
                        for n_shots in num_shots:
                            labels = np.load(labels_path.format(
                                root_directory=root_directory,
                                model=model,
                                seed=seed,
                                dataset=dataset,
                                num_shots=n_shots
                            ))
                            if method == "no_adaptation":
                                logits = np.load(logits_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots,
                                    method="logits",
                                    num_samples="original"
                                ))
                            else:
                                logits = np.load(logits_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots,
                                    method=method,
                                    num_samples=n_samples
                                ))
                            for i, bi in enumerate(boots_idx):
                                results["num_shots"].append(n_shots)
                                results["seed"].append(seed)
                                results["bootstrap_index"].append(i)
                                for metric in metrics:
                                    score = compute_metric(logits, labels, metric, bootstrap_idx=bi)
                                    results[metric].append(score)
                    df = pd.DataFrame(results)
                    for m, metric in enumerate(metrics):
                        metric_vs_n_shots = df.groupby("num_shots").agg({metric: ["mean", "std"]})
                        means = metric_vs_n_shots.loc[num_shots,(metric,"mean")].values
                        stds = metric_vs_n_shots.loc[num_shots,(metric,"std")].values
                        axes[m,d].plot(num_shots,means,**methods[method])
                        axes[m,d].fill_between(num_shots, means-stds, means+stds, alpha=0.2,color=methods[method]["color"])
                        axes[m,d].grid(True)
                        axes[m,d].set_xticks(num_shots)
                        axes[m,d].set_xticklabels(num_shots, fontsize=12)
                        axes[m,d].set_xlim(num_shots[0],num_shots[-1])
                        if m == 0:
                            axes[m,d].set_title(datasets[dataset], fontsize=18)
                        if d == 0:
                            axes[m,d].set_ylabel(metrics[metric], fontsize=18)
            handles, labels = axes[m,d].get_legend_handles_labels()
            first_handle = handles.pop(0)
            handles.append(first_handle)
            first_label = labels.pop(0)
            labels.append(first_label)
            fig.legend(handles, labels, loc='upper center', ncol=len(methods), fontsize=18)
            fig.supxlabel("Number of shots", fontsize=20)
            fig.savefig(f"results/plot/{id}-model_{model}-n_samples_{n_samples}")


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