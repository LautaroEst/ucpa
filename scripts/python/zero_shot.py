

import argparse
import json
import os
import numpy as np
import torch
from ucpa.data import ClassificationDatasetDict, SequentialLoaderWithDataCollator
from ucpa.models import load_base_model, LanguageModelClassifier
from tqdm import tqdm


def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
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

    # Parse command line arguments and read config file
    args = parse_args()

    # Instantiate base model
    print("Loading base model...")
    base_model, tokenizer = load_base_model(args.model)
    model = LanguageModelClassifier(
        base_model=base_model,
        tokenizer=tokenizer
    )

    for config in args.configs_dicts:

        model.set_labels_names(config["labels"])

        # Instantiate classification model and dataset
        dataset = ClassificationDatasetDict(
            args.dataset,
            data_dir=os.path.join(args.root_directory,"data"),
            template=config["prompt"],
            num_train_samples=config["num_train_samples"],
            num_test_samples=config["num_test_samples"],
            random_state=args.seed,
            sort_by_length=True,
            ascending=False
        )
        results = {
            "train": run_model(model, dataset["train"], tokenizer, config["batch_size"]),
            "test": run_model(model, dataset["test"], tokenizer, config["batch_size"])
        } 
    
        # Save results
        for key, result in results.items():
            dir_name = os.path.join(args.root_directory, "results", args.dataset, args.model, args.seed)
            for output in result.keys():
                np.save(os.path.join(dir_name,f"{key}.{output}.npy"),result[output])


def run_model(model, dataset, tokenizer, batch_size = 32):
    loader = SequentialLoaderWithDataCollator(dataset, tokenizer, batch_size)

    all_logits = []
    all_labels = []
    train_bar = tqdm(loader, desc="Batches", total=len(loader))
    for batch in train_bar:
        with torch.no_grad():
            _, logits = model(batch["sentences"])
            # logprobs = torch.log_softmax(logits,dim=-1).cpu().numpy()
            logits = logits.cpu().numpy()
        all_logits.append(logits)
        all_labels.append(batch["labels"])
    all_logits = np.concatenate(all_logits,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)

    return {
        "logits": all_logits, 
        "labels": all_labels
    }

           
        

if __name__ == "__main__":
    main()