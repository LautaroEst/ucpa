


import json
import os
import sys
sys.path.append("../efficient-reestimation")
from src.utils import get_results_ids_from_config
import pickle
import numpy as np


def main():
    
    seeds = [82033, 12782, 1263, 987, 12299, 9203, 4, 20343, 43, 92374]
    for seed in seeds:
        for dataset in ["trec", "sst2"]:
            for n_shot in [0, 1, 4, 8]:
                os.makedirs(f"results/model_calibration/gpt2-xl/{seed}/{dataset}_{n_shot}_shot", exist_ok=True)
    
    with open(os.path.join("../efficient-reestimation/configs/models/gpt2-xl_trec_sst2.json")) as experiment_file:
        experiment_config = json.load(experiment_file)
    results_ids = get_results_ids_from_config("../efficient-reestimation", experiment_config)

    for result_id in results_ids:
        with open(f"../efficient-reestimation/results/train_test/{result_id}/config.json", "r") as f:
            original_config = json.load(f)
        for seed in seeds:
            if len(os.listdir(f"results/model_calibration/gpt2-xl/{seed}/{original_config['dataset']}_{original_config['n_shots']}_shot")) == 0:
                break
        root_save_path = f"results/model_calibration/gpt2-xl/{seed}/{original_config['dataset']}_{original_config['n_shots']}_shot"
        with open(os.path.join(f"../efficient-reestimation/results/train_test/{result_id}/train.pkl"), "rb") as f:
            train_results = pickle.load(f)
            train_logits = np.log(train_results["train_probs"])
            train_labels = train_results["train_labels"]
            np.save(os.path.join(root_save_path,f"train.logits.npy"),train_logits)
            np.save(os.path.join(root_save_path,f"train.labels.npy"),train_labels)
        with open(os.path.join(f"../efficient-reestimation/results/train_test/{result_id}/test.pkl"), "rb") as f:
            test_results = pickle.load(f)
            test_logits = np.log(test_results["test_probs"])
            test_labels = test_results["test_labels"]
            np.save(os.path.join(root_save_path,f"test.logits.npy"),test_logits)
            np.save(os.path.join(root_save_path,f"test.labels.npy"),test_labels)





if __name__ == "__main__":
    main()