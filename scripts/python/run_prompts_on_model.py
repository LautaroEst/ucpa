

import argparse
from copy import deepcopy
from itertools import chain
import json
import os
import numpy as np
import torch
from ucpa.data import load_dataset, SequentialLoaderWithDataCollator, ClassificationDataSplit, load_template
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

    # Run only valid configs:
    valid_configs = []
    for config in args.configs_dicts:
        if config["type"] == "run_on_dataset":
            if any([not os.path.exists(os.path.join(root_save_path, config["id"], f"{split}.{output}.npy")) for output in output_keys for split in splits]):
                valid_configs.append(config)
        elif config["type"] == "run_on_predefined_prompt":
            if any([not os.path.exists(os.path.join(root_save_path, config["id"], f"predefined.{output}.npy")) for output in output_keys]):
                valid_configs.append(config)
        else:
            raise ValueError(f"Config type {config['type']} not supported.")
        
    if len(valid_configs) == 0:
        print("Everything computed for this model. Skipping...")
        return

    # Instantiate base model
    print("Loading base model...")
    model = LanguageModelClassifier(args.model.replace("--","/"))

    for config in valid_configs:

        # Results path for this config
        os.makedirs(os.path.join(root_save_path, config["id"]), exist_ok=True)

        if config["type"] == "run_on_dataset":
            run_model_on_dataset(
                data_directory=os.path.join(args.root_directory, "data"), 
                model=model, 
                dataset_name=config["dataset"],
                num_train_samples=args.num_train_samples if args.num_train_samples != -1 else None,
                num_test_samples=args.num_test_samples if args.num_test_samples != -1 else None,
                labels=config["labels"], 
                template_args=config["template_args"],
                results_directory=os.path.join(root_save_path, config["id"]),
                batch_size=config["batch_size"],
                random_state=seed+int(config["id-num"])
            )
        elif config["type"] == "run_on_predefined_prompt":
            run_model_on_prompt(
                model=model,
                queries=config["queries"],
                labels=config["labels"],
                template_args=config["template_args"],
                results_directory=os.path.join(root_save_path, config["id"]),
                batch_size=config["batch_size"],
                random_state=seed+int(config["id-num"])
            )
        
        with open(os.path.join(root_save_path, config["id"], "config.json"), "w") as f:
            json.dump(config,f)


def run_model_on_dataset(data_directory, model, dataset_name, num_train_samples, num_test_samples, labels, template_args, results_directory, batch_size = 32, random_state = None):

    # Instantiate classification model and dataset
    dataset = load_dataset(
        dataset_name,
        data_dir=data_directory,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        random_state=random_state,
        sort_by_length=True,
        ascending=False,
        template_args=template_args
    )

    # Save results
    for split in splits:
        if all([os.path.exists(os.path.join(results_directory, f"{split}.{output}.npy")) for output in output_keys]):
            continue
        
        # Create loader and trainer to predict
        loader = SequentialLoaderWithDataCollator(dataset[split], model.tokenizer, labels, batch_size)
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=-1,
            enable_checkpointing=False, 
            logger=False
        )
        predictions = trainer.predict(model, loader)
        ids, prompts, logits, labels_arr = zip(*predictions)
        
        result = {
            "ids": np.concatenate(ids),
            "prompts": np.concatenate(prompts),
            "logits": np.concatenate(logits),
            "labels": np.concatenate(labels_arr)
        }

        for output in output_keys:
            np.save(os.path.join(results_directory, f"{split}.{output}.npy"), result[output])

def run_model_on_prompt(model, queries, labels, template_args, results_directory, batch_size = 32, random_state = None):
    
    template_args = deepcopy(template_args)
    dataset = ClassificationDataSplit(
        ids=list(range(len(queries))), 
        sentences=queries,
        labels=labels,
        template=load_template(**template_args),
        sort_by_length=False
    )
    loader = SequentialLoaderWithDataCollator(dataset, model.tokenizer, labels, batch_size)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        enable_checkpointing=False, 
        logger=False
    )
    predictions = trainer.predict(model, loader)
    ids, prompts, logits, labels_arr = zip(*predictions)
    result = {
        "ids": np.concatenate(ids),
        "prompts": np.concatenate(prompts),
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels_arr)
    }
    for output in output_keys:
        np.save(os.path.join(results_directory, f"train.{output}.npy"), result[output])

if __name__ == "__main__":
    main()