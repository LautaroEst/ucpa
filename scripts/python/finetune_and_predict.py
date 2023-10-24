import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm


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
    seeds = [int(s) for s in args.seeds.split(" ")]

    for seed in seeds:
        
        # Root path to save results:
        root_save_path = os.path.join(args.root_directory, "results", args.experiment_name, args.model, str(seed))
        
        for config in args.configs_dicts:

            # Results path for this config
            os.makedirs(os.path.join(root_save_path, config["id"]), exist_ok=True)
            
            # Random state
            rs = np.random.RandomState(seed+config["random_state"])

            # Load logits and labels
            train_prompts = np.load(os.path.join(config["train"].format(root_directory=args.root_directory,seed=seed),"prompts.npy"))
            train_labels = np.load(os.path.join(config["train"].format(root_directory=args.root_directory,seed=seed),"labels.npy"))
            test_prompts = np.load(os.path.join(config["test"].format(root_directory=args.root_directory,seed=seed),"prompts.npy"))
            test_labels = np.load(os.path.join(config["test"].format(root_directory=args.root_directory,seed=seed),"labels.npy"))
            
            # Save them in the new results directory
            np.save(os.path.join(root_save_path,str(config["id"]),f"train.labels.original.npy"),train_labels)
            np.save(os.path.join(root_save_path,str(config["id"]),f"test.labels.original.npy"),test_labels)

            # Train samples:
            if config["num_train_samples"] is None:
                train_samples = tqdm([len(train_labels)])
            else:
                train_samples = tqdm(config["num_train_samples"])
            for n_samples in train_samples:
                hyperparams = config["num_train_samples"][n_samples]
                n_samples = int(n_samples)
                train_idx = rs.choice(len(train_labels), n_samples, replace=False)
                sub_train_prompts = train_prompts[train_idx,:].copy()
                sub_train_labels = train_labels[train_idx].copy()
                train_samples.set_description(desc=f"{n_samples} samples")
                cal_test_logits = run_finetunning(sub_train_prompts, sub_train_labels, test_prompts, model=args.model, **hyperparams)
                np.save(os.path.join(root_save_path,str(config["id"]),f"test.finetuned.{n_samples}.npy"),cal_test_logits)
                np.save(os.path.join(root_save_path,str(config["id"]),f"train.idx.{n_samples}.npy"),train_idx)
                train_samples.set_description(desc=config["id"])
            
            # Save config
            with open(os.path.join(root_save_path, config["id"], "config.json"), "w") as f:
                json.dump(config,f,indent=4)


def run_finetunning(train_prompts, train_labels, test_prompts, model="gpt2", batch_size=32, epochs=3):

    test_logits = torch.from_numpy(test_logits)
    train_labels = torch.from_numpy(train_labels)

    print("Loading base model to train...")
    base_model = LanguageModel(base_model=model)
    train_loader = PlainTextLoader(train_prompts, base_model.tokenizer, batch_size=batch_size)
    val_loader = PlainTextLoader(train_prompts, base_model.tokenizer, batch_size=batch_size)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        enable_checkpointing=False, 
        logger=False,
        max_epochs=epochs
    )
    trainer.fit(base_model, train_loader, val_loader)

    print("Predicting on test with finetuned model...")
    model = LanguageModelClassifier.from_base_model(base_model, model_name=model)
    test_loader = TextClassificationLoaderFromPrompts(test_prompts, -np.ones(len(test_prompts)), model.tokenizer)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        enable_checkpointing=False, 
        logger=False
    )
    predictions = trainer.predict(model, test_loader)
    _, _, cal_logits, _ = zip(*predictions)
    return cal_logits


if __name__ == "__main__":
    main()