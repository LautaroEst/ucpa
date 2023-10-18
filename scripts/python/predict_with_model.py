import argparse
import json
import os
import numpy as np
import torch
from ucpa.data import load_dataset, load_template, SequentialLoaderWithDataCollator
from ucpa.models import LanguageModelClassifier
import lightning.pytorch as pl


def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    with open(os.path.join(args.root_directory,f"configs/{args.experiment_name}/{args.model}.json"), "r") as f:
        configs_dicts = json.load(f)
    setattr(args,"configs_dicts",configs_dicts)
    return args


def main():

    # Parse command line arguments and define save path
    args = parse_args()
    data_dir = os.path.join(args.root_directory, "data")
    seeds = [int(s) for s in args.seeds.split(" ")]

    # Instantiate base model
    print("Loading base model...")
    model = LanguageModelClassifier(args.model.replace("--","/"))

    for seed in seeds:
        
        # Root path to save results:
        root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.model, str(seed))
        
        # Run only valid configs:
        valid_configs = [config for config in args.configs_dicts if not os.path.exists(os.path.join(root_save_path, config["id"], f"logits.npy"))]
        if len(valid_configs) == 0:
            print("Everything computed for this seed. Skipping...")
            continue
        
        for config in valid_configs:

            # Results path for this config
            os.makedirs(os.path.join(root_save_path, config["id"]), exist_ok=True)
            
            # Create template
            if config["template_args"]["name"] == "preface_plus_shots":
                if "sample_shots_from" in config["template_args"]:
                    config["template_args"]["sample_shots_from"]["data_dir"] = data_dir
                    config["template_args"]["sample_shots_from"]["random_state"] = seed+config["random_state"]
            template = load_template(**config["template_args"])

            # Create dataset
            dataset = load_dataset(
                name=config["dataset"],
                data_dir=data_dir,
                split=config["split"],
                template=template,
                num_samples=config["num_samples"],
                sort_by_length=config["split"] == "test",
                ascending=False, 
                random_state=seed+config["random_state"]
            )

            # Run predictions
            run_model_on_dataset(
                model=model,
                dataset=dataset,
                labels=config["labels"], 
                results_directory=os.path.join(root_save_path, config["id"]),
                batch_size=config["batch_size"]
            )
            
            # Save config
            with open(os.path.join(root_save_path, config["id"], "config.json"), "w") as f:
                json.dump(config,f,indent=4)



def run_model_on_dataset(model, dataset, labels, results_directory, batch_size = 32):

    # Create loader and trainer to predict
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
        "original_ids": np.concatenate(ids),
        "prompts": np.concatenate(prompts),
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels_arr)
    }

    for output in result:
        np.save(os.path.join(results_directory, f"{output}.npy"), result[output])


if __name__ == "__main__":
    main()