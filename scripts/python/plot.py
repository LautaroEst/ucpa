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
    results = {
        "num_samples": [], "seed": [], "bootstrap_index": [],
        **{metric: [] for metric in metrics}
    }
    rs = np.random.RandomState(random_state)
    for model in models:
        for n_shots in num_shots:
            fig, axes = plt.subplots(len(metrics),len(datasets),figsize=(20,5))
            for d, (ax, dataset) in enumerate(zip(axes,datasets)):
                N = len(labels_path.format(
                    root_directory=root_directory,
                    model=model,
                    seed=seeds[0],
                    dataset=dataset,
                    num_shots=n_shots
                ))
                boots_idx = [rs.choice(N, N, replace=True) for _ in range(bootstrap)] if bootstrap is not None else [None]
                for method in methods:
                    for m, metric in enumerate(metrics):
                        for n_samples in num_samples:
                            for seed in seeds:
                                logits = logits_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots,
                                    method=method,
                                    num_samples=n_samples
                                )
                                labels = labels_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots
                                )
                                for i, bi in enumerate(boots_idx):
                                    score = compute_metric(logits, labels, metric, bootstrap_idx=bi)
                                    results["num_samples"].append(n_samples)
                                    results["seed"].append(seed)
                                    results["bootstrap_index"].append(i)
                                    results[f"metric:{metric}"].append(score)
                            df = pd.DataFrame(results)
                            metric_vs_n_samples = df.groupby("num_samples").agg({metric: ["mean", "std"]})
                            means = metric_vs_n_samples.loc[num_samples,(metric,"mean")]
                            stds = metric_vs_n_samples.loc[num_samples,(metric,"std")]
                            ax[m].plot(num_samples,means,**methods[method])
                            ax[m].fill_between(num_shots, means-stds, means+stds, alpha=0.2,**methods[method])
                        ax[m].grid(True)
                        ax[m].set_xscale("log")
                        ax[m].set_xticks(num_samples)
                        ax[m].set_xticklabels(num_samples, fontsize=12)
                        ax[m].set_xlim(num_samples[0],num_samples[-1])
                        if m == 0:
                            ax[m].set_title(datasets[dataset], fontsize=18)
                        if d == 0:
                            ax[m].set_ylabel(metrics[metric], fontsize=18)


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
    results = {
        "num_shots": [], "seed": [], "bootstrap_index": [],
        **{metric: [] for metric in metrics}
    }
    rs = np.random.RandomState(random_state)
    for model in models:
        for n_samples in num_samples:
            fig, axes = plt.subplots(len(metrics),len(datasets),figsize=(20,5))
            for d, (ax, dataset) in enumerate(zip(axes,datasets)):
                N = len(labels_path.format(
                    root_directory=root_directory,
                    model=model,
                    seed=seeds[0],
                    dataset=dataset,
                    num_shots=num_shots[0]
                ))
                boots_idx = [rs.choice(N, N, replace=True) for _ in range(bootstrap)] if bootstrap is not None else [None]
                for method in methods:
                    for m, metric in enumerate(metrics):
                        for n_shots in num_shots:            
                            for seed in seeds:
                                logits = logits_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots,
                                    method=method,
                                    num_samples=n_samples
                                )
                                labels = labels_path.format(
                                    root_directory=root_directory,
                                    model=model,
                                    seed=seed,
                                    dataset=dataset,
                                    num_shots=n_shots
                                )
                                for i, bi in enumerate(boots_idx):
                                    score = compute_metric(logits, labels, metric, bootstrap_idx=bi)
                                    results["num_shots"].append(n_shots)
                                    results["seed"].append(seed)
                                    results["bootstrap_index"].append(i)
                                    results[f"metric:{metric}"].append(score)
                            df = pd.DataFrame(results)
                            metric_vs_n_shots = df.groupby("num_shots").agg({metric: ["mean", "std"]})
                            means = metric_vs_n_shots.loc[num_shots,(metric,"mean")]
                            stds = metric_vs_n_shots.loc[num_shots,(metric,"std")]
                            ax[m].plot(num_shots,means,**methods[method])
                            ax[m].fill_between(num_shots, means-stds, means+stds, alpha=0.2,**methods[method])
                        ax[m].grid(True)
                        ax[m].set_xticks(num_shots)
                        ax[m].set_xticklabels(num_shots, fontsize=12)
                        ax[m].set_xlim(num_shots[0],num_shots[-1])
                        if m == 0:
                            ax[m].set_title(datasets[dataset], fontsize=18)
                        if d == 0:
                            ax[m].set_ylabel(metrics[metric], fontsize=18)


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