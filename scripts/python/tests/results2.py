
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def main():
    models = ["gpt2-xl"]
    datasets = ["tony_zhao_trec"]#,"tony_zhao_sst2","tony_zhao_agnews","tony_zhao_dbpedia"]
    metrics = ["norm_cross_entropy", "accuracy"]
    seeds = ["82033", "12782", "1263", "987", "12299", "9203", "4", "20343", "43", "92374"]

    print("Collecting results...")
    results = collect_results(".", "paper_results", models, datasets, metrics, seeds)
    import pdb; pdb.set_trace()


def collect_results(root_directory, experiment_name, models, datasets, metrics, seeds):
    results = []
    for model in models:
        for dataset in datasets:
            results_dir = os.path.join(root_directory, "results", experiment_name, dataset, model)
            for seed in seeds:
                for n_shot in os.listdir(os.path.join(results_dir,seed)):
                    labels = np.load(os.path.join(results_dir,seed,n_shot,"test.labels.npy"))
                    original_logits = np.load(os.path.join(results_dir,seed,n_shot,"test.logits.npy"))
                    results_dict = {f"metric:{metric}": compute_metric(original_logits, labels, metric) for metric in metrics}
                    results_dict["model"] = model
                    results_dict["dataset"] = dataset
                    results_dict["method"] = "no_adaptation"
                    results_dict["num_shots"] = int(n_shot.split("_")[0])
                    results_dict["num_samples"] = -1
                    results_dict["random_state"] = int(seed)
                    results.append(results_dict)
                
                    for result in os.listdir(os.path.join(results_dir,seed,n_shot,"calibration")):
                        method, num_samples = result.split(".")[1:3]
                        result_logits = np.load(os.path.join(results_dir,seed,n_shot,"calibration",result))
                        results_dict = {f"metric:{metric}": compute_metric(result_logits, labels, metric) for metric in metrics}
                        results_dict["model"] = model
                        results_dict["dataset"] = dataset
                        results_dict["method"] = method
                        results_dict["num_shots"] = int(n_shot.split("_")[0])
                        results_dict["num_samples"] = int(num_samples)
                        results_dict["random_state"] = int(seed)
                        results.append(results_dict)

    results = pd.DataFrame.from_records(results)
    results = results.groupby(by=["model","dataset","method","num_shots","num_samples"]).agg({f"metric:{metric}": ["mean","std"] for metric in metrics})
    return results


def compute_metric(logits, labels, metric="cross_entropy"):
    
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