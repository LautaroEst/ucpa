

import argparse
from itertools import chain
import json
import os
import numpy as np
import torch
from ucpa.data import load_dataset, SequentialLoaderWithDataCollator
from ucpa.models import LanguageModelClassifier
import lightning.pytorch as pl

output_keys = ["ids", "prompts", "logits", "labels"]
splits = ["train", "test"]

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--num_train_samples", type=int, default=1000)
    parser.add_argument("--num_test_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.root_directory,f"configs/{args.experiment_name}/{args.model}.jsonl"), "r") as f:
        configs_dicts = [json.loads(config) for config in f.read().splitlines()]
    setattr(args,"configs_dicts",configs_dicts)
    
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()
    seed = args.seed
    root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.model, str(seed))

    valid_configs = [config for config in args.configs_dicts if any([not os.path.exists(os.path.join(root_save_path,config["id"], f"{split}.{output}.npy")) for output in output_keys for split in splits])]
    if len(valid_configs) == 0:
        print("Everything computed for this model. Skipping...")
        return

    # Instantiate base model
    print("Loading base model...")
    model = LanguageModelClassifier(args.model.replace("--","/"))

    for config in valid_configs:

        # Results path for this config
        os.makedirs(os.path.join(root_save_path, config["id"]), exist_ok=True)

        # Instantiate classification model and dataset
        dataset = load_dataset(
            config["dataset"],
            n_shots=config["n_shots"],
            data_dir=os.path.join(args.root_directory, "data"),
            num_train_samples=args.num_train_samples if args.num_train_samples != -1 else None,
            num_test_samples=args.num_test_samples if args.num_test_samples != -1 else None,
            random_state=seed+int(config["id-num"]),
            sort_by_length=True,
            ascending=False,
            template_args=config["template_args"]
        )

        # Save results
        for split in splits:
            if all([os.path.exists(os.path.join(root_save_path, config["id"], f"{split}.{output}.npy")) for output in output_keys]):
                continue
            result = run_model(model, dataset[split], config["labels"], config["batch_size"])
            for output in output_keys:
                np.save(os.path.join(root_save_path, config["id"], f"{split}.{output}.npy"), result[output])
        with open(os.path.join(root_save_path, config["id"], "config.json"), "w") as f:
            json.dump(config,f)


def run_model(model, dataset, labels, batch_size = 32):

    # Create loader and trainer to predict
    loader = SequentialLoaderWithDataCollator(dataset, model.tokenizer, labels, batch_size)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        enable_checkpointing=False, 
        logger=False
    )
    predictions = trainer.predict(model, loader)
    ids, prompts, logits, labels = zip(*predictions)
    return {
        "ids": np.concatenate(ids),
        "prompts": np.concatenate(prompts),
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels)
    }

    

if __name__ == "__main__":
    main()