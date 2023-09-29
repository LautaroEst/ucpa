
import argparse
from ucpa.data.templates import PrefacePlusShotsPrompt
from ucpa.models import LanguageModelClassifier
from torch.utils.data import Dataset
from ucpa.data.loaders import SequentialLoaderWithDataCollator
import lightning.pytorch as pl
import torch
import numpy as np

preface = ""
query_prefix = "Review:"
label_prefix = "Sentiment:"
prefix_sample_separator = " "
query_label_separator = "\n"
shot_separator = "\n\n"

labels_idx2name = {0: "Negative", 1: "Positive"}

shots = {
    "prompt_shots_sentences": [
        "what a pity ... that the material is so second-rate .", 
        "does not go far enough in its humor or stock ideas to stand out as particularly memorable or even all that funny .", 
        "leaves viewers out in the cold and undermines some phenomenal performances .", 
        "an engrossing story that combines psychological drama , sociological reflection , and high-octane thriller ."
    ], 
    "prompt_shots_labels": [0, 0, 0, 1]
}

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    args = parser.parse_args()
    return args


class DummyDataset(Dataset):

    query = "if you are curious to see the darker side of what 's going on with young tv actors -lrb- dawson leery did what ?!? -rrb- , or see some interesting storytelling devices , you might want to check it out , but there 's nothing very attractive about this movie ."

    def __init__(self, template):
        self.template = template

    def __getitem__(self, idx):
        return {
            "id": idx, 
            "sentence": self.query, 
            "label": 0
        }
    def __len__(self):
        return 8
    


def main():

    args = parse_args()

    model = LanguageModelClassifier("gpt2-xl")

    template = PrefacePlusShotsPrompt(
        preface=preface,
        sentences_shots=shots["prompt_shots_sentences"],
        labels_shots=[labels_idx2name[i] for i in shots["prompt_shots_labels"]],
        query_prefix=query_prefix,
        label_prefix=label_prefix,
        prefix_sample_separator=prefix_sample_separator,
        query_label_separator=query_label_separator,
        shot_separator=shot_separator
    )
    dataset = DummyDataset(template)
    loader = SequentialLoaderWithDataCollator(dataset, model.tokenizer, [labels_idx2name[i] for i in range(len(labels_idx2name))], batch_size=4)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        enable_checkpointing=False, 
        logger=False
    )
    predictions = trainer.predict(model, loader)
    logits, labels = zip(*predictions)
    print(np.exp(logits))
    print(labels)


if __name__ == "__main__":
    main()