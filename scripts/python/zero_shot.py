

import argparse
import json
import os
import numpy as np
import torch
from ucpa.data import load_dataset, SequentialLoaderWithDataCollator, SimpleQuerySubstitutionPrompt
from ucpa.models import LanguageModelClassifier
import lightning.pytorch as pl


def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs_dicts = [json.loads(config) for config in f.read().splitlines()]
    setattr(args,"configs_dicts",configs_dicts)
    
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()
    root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.dataset, args.model, str(args.seed))

    # Instantiate base model
    print("Loading base model...")
    model = LanguageModelClassifier(args.model)

    for config in args.configs_dicts:

        # Instantiate classification model and dataset
        dataset = load_dataset(
            args.dataset,
            data_dir=os.path.join(args.root_directory,"data"),
            template=SimpleQuerySubstitutionPrompt(config["prompt"], "{query}"),
            num_train_samples=config["num_train_samples"],
            num_test_samples=config["num_test_samples"],
            random_state=args.seed,
            sort_by_length=True,
            ascending=False
        )

        results = {
            "train": run_model(model, dataset["train"], config["labels"], config["batch_size"]),
            "test": run_model(model, dataset["test"], config["labels"], config["batch_size"])
        } 

        # Save results
        for key, result in results.items():
            for output in result.keys():
                np.save(os.path.join(root_save_path,str(config["id"]),f"{key}.{output}.npy"),result[output])


def run_model(model, dataset, labels, batch_size = 32):

    # Create loader and trainer to predict
    loader = SequentialLoaderWithDataCollator(dataset, model.tokenizer, labels, batch_size)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1
    )
    predictions = trainer.predict(model, loader)
    default_lightning_logs_dir = os.path.join(os.getcwd(),"lightning_logs")
    if os.path.exists(default_lightning_logs_dir):
        os.removedirs(default_lightning_logs_dir)
    logits, labels = zip(*predictions)
    return {
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels)
    }

    

if __name__ == "__main__":
    main()