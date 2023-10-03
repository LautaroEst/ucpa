

import argparse
import json
from ucpa.data.templates import PrefacePlusShotsPrompt
from ucpa.models import LanguageModelClassifier
from torch.utils.data import Dataset
from ucpa.data.loaders import SequentialLoaderWithDataCollator
import lightning.pytorch as pl
import torch
import numpy as np
import pickle


preface = ""
query_prefix = "Review:"
label_prefix = "Sentiment:"
prefix_sample_separator = " "
query_label_separator = "\n"
shot_separator = "\n\n"

labels_idx2name = {0: "Negative", 1: "Positive"}


def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    args = parser.parse_args()
    return args


class DummyDataset(Dataset):

    def __init__(self, template, queries):
        self.template = template
        self.queries = queries

    def __getitem__(self, idx):
        return {
            "id": idx, 
            "sentence": self.queries[idx], 
            "label": 0
        }
    def __len__(self):
        return len(self.queries)
    

def main():

    args = parse_args()

    # "dataset": "sst2", "model": "gpt2-xl", "n_shots": 1, "random_state": 58776
    original_results_id = "31323818017343297470302527315948969317"
    with open(f"../efficient-reestimation/results/train_test/{original_results_id}/test.pkl", "rb") as f:
        original_results = pickle.load(f)
    with open(f"../efficient-reestimation/results/train_test/{original_results_id}/prompt_shots.json", "r") as f:
        original_prompt_shots = json.load(f)

    template = PrefacePlusShotsPrompt(
        preface=preface,
        sentences_shots=original_prompt_shots["prompt_shots_sentences"],
        labels_shots=[labels_idx2name[i] for i in original_prompt_shots["prompt_shots_labels"]],
        query_prefix=query_prefix,
        label_prefix=label_prefix,
        prefix_sample_separator=prefix_sample_separator,
        query_label_separator=query_label_separator,
        shot_separator=shot_separator
    )
    dataset = DummyDataset(template,original_results["test_queries"])
    model = LanguageModelClassifier("gpt2-xl")
    loader = SequentialLoaderWithDataCollator(dataset, model.tokenizer, [labels_idx2name[i] for i in range(len(labels_idx2name))], batch_size=8)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        enable_checkpointing=False, 
        logger=False
    )
    predictions = trainer.predict(model, loader)
    ids, prompts, logits, labels = zip(*predictions)

    logits = np.concatenate(logits)
    test_probs_new = torch.softmax(torch.from_numpy(logits),dim=1).numpy()
    test_probs_original = original_results["test_probs"] / original_results["test_probs"].sum(axis=1,keepdims=True)

    print(np.sum(np.abs(test_probs_new - test_probs_original) > 1e-3))
    

if __name__ == "__main__":
    main()