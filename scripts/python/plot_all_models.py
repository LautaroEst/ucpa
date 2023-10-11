
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--metric", type=str, default="")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--seeds", type=str)
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
    seeds = [int(s) for s in args.seeds.split(" ")]
    calibration_results_dir = os.path.join(args.root_directory, "results/model_calibration")

    # Number of samples vs metric
    print("Plotting methods vs metric for all models...")
    plot_results(
        args.root_directory, 
        args.experiment_name,
        calibration_results_dir,
        models, 
        args.metric,
        args.num_shots,
        seeds
    )

    print("Plotting methods vs metric for all models (relative)...")
    plot_relative_results(
        args.root_directory, 
        args.experiment_name,
        calibration_results_dir,
        models, 
        args.metric,
        args.num_shots,
        seeds
    )
        
def plot_results(root_directory, experiment_name, calibration_results_dir, models, metric, num_shots, seeds):
    all_results = []
    for model in models:
        results = pd.read_csv(os.path.join(calibration_results_dir,model,"results.csv"))
        results = results[results["num_samples"] != 10].reset_index(drop=True)
        results["model"] = model
        results = results.drop(columns=[col for col in results.columns if "metric:" in col and col != f"metric:{metric}"]).reset_index(drop=True)
        results = results[results["random_state"].isin(seeds)].reset_index(drop=True)
        results = results[results["num_shots"] == num_shots].reset_index(drop=True)
        results = results.drop(columns=["bootstrap_iter"]).reset_index(drop=True)
        results = results[results["method"].isin(["UCPA","SUCPA","no_adaptation"])].reset_index(drop=True)
        no_adaptation = results.loc[results["method"] == "no_adaptation", :].copy()
        results = results.loc[results["method"] != "no_adaptation", :]
        results_with_no_adaptation = [results]
        for num_samples in results["num_samples"].unique():
            no_adaptation["num_samples"] = num_samples
            results_with_no_adaptation.append(no_adaptation.copy())
        results = pd.concat(results_with_no_adaptation,axis=0)
        all_results.append(results)
    all_results = pd.concat(all_results,axis=0)

    num_samples = all_results["num_samples"].unique()
    datasets = all_results["dataset"].unique()
    for n in num_samples:
        fig, axes = plt.subplots(1,len(datasets),figsize=(20,5))
        for ax, dataset in zip(axes,datasets):
            dataset_results = all_results[(all_results["dataset"] == dataset) & (all_results["num_samples"] == n)].reset_index(drop=True)
            sns.boxplot(data=dataset_results,x="model",y=f"metric:{metric}",hue="method",ax=ax)
            ax.set_title(dataset_short2name[dataset])
            ax.grid(True)
            ax.get_legend().set_visible(False)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45,fontsize=12,ha="right")
        axes[0].set_ylabel(metric_short2name[metric])
        handles, labels = ax.get_legend_handles_labels()
        first_handle = handles.pop(0)
        handles.append(first_handle)
        first_label = labels.pop(0)
        labels.append(first_label)
        fig.suptitle(f"Results for {n} training samples", fontsize=14)
        fig.legend(handles, labels, loc='center right', fontsize=12)
        # fig.supxlabel("Number of training samples", fontsize=12)
        # fig.autofmt_xdate(rotation=45)
        fig.savefig(os.path.join(root_directory,"results",experiment_name,f"results_{n}-samples_{num_shots}-shots.png"),bbox_inches = 'tight')


def plot_relative_results(root_directory, experiment_name, calibration_results_dir, models, metric, num_shots, seeds):
    all_results = []
    for model in models:
        results = pd.read_csv(os.path.join(calibration_results_dir,model,"results.csv"))
        results = results[results["num_samples"] != 10].reset_index(drop=True)
        results["model"] = model
        results = results.drop(columns=[col for col in results.columns if "metric:" in col and col != f"metric:{metric}"]).reset_index(drop=True)
        results = results[results["random_state"].isin(seeds)].reset_index(drop=True)
        results = results[results["num_shots"] == num_shots].reset_index(drop=True)
        results = results.drop(columns=["bootstrap_iter"]).reset_index(drop=True)
        results = results[results["method"].isin(["UCPA","SUCPA","no_adaptation"])].reset_index(drop=True)
        no_adaptation = results.loc[results["method"] == "no_adaptation", :].copy()
        results = results.loc[results["method"] != "no_adaptation", :]
        results_with_no_adaptation = [results]
        for num_samples in results["num_samples"].unique():
            no_adaptation["num_samples"] = num_samples
            results_with_no_adaptation.append(no_adaptation.copy())
        results = pd.concat(results_with_no_adaptation,axis=0)
        all_results.append(results)
    all_results = pd.concat(all_results,axis=0)

    num_samples = all_results["num_samples"].unique()
    datasets = all_results["dataset"].unique()
    for n in num_samples:
        fig, axes = plt.subplots(1,len(datasets),figsize=(20,5))
        for ax, dataset in zip(axes,datasets):
            dataset_results = all_results[(all_results["dataset"] == dataset) & (all_results["num_samples"] == n)].reset_index(drop=True)
            for method in ["UCPA", "SUCPA"]:
                dataset_results.loc[dataset_results["method"] == method,f"metric:{metric}"] = (dataset_results.loc[dataset_results["method"] == "no_adaptation",f"metric:{metric}"] - dataset_results.loc[dataset_results["method"] == method,f"metric:{metric}"]) / dataset_results.loc[dataset_results["method"] == method,f"metric:{metric}"] * 100
            import pdb; pdb.set_trace()
            dataset_results = dataset_results.loc[dataset_results["method"] != "no_adaptation",:]
            sns.boxplot(data=dataset_results,x="model",y=f"metric:{metric}",hue="method",ax=ax)
            ax.set_title(dataset_short2name[dataset])
            ax.grid(True)
            ax.get_legend().set_visible(False)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45,fontsize=12,ha="right")
        axes[0].set_ylabel(metric_short2name[metric])
        handles, labels = ax.get_legend_handles_labels()
        first_handle = handles.pop(0)
        handles.append(first_handle)
        first_label = labels.pop(0)
        labels.append(first_label)
        fig.suptitle(f"Results for {n} training samples", fontsize=14)
        fig.legend(handles, labels, loc='center right', fontsize=12)
        # fig.supxlabel("Number of training samples", fontsize=12)
        # fig.autofmt_xdate(rotation=45)
        fig.savefig(os.path.join(root_directory,"results",experiment_name,f"relative_results_{n}-samples_{num_shots}-shots.png"),bbox_inches = 'tight')



if __name__ == "__main__":
    main()