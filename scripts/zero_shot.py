

import os
import numpy as np
import torch
from ucpa.utils import parse_args, read_config, save_config
from ucpa.data import PromptTemplate, ClassificationDatasetDict, SequentialLoaderWithDataCollator
from ucpa.models import load_base_model, FewShotLanguageModelClassifier
from tqdm import tqdm

PREFIX_SAMPLE_SEPARATOR = " "
QUERY_LABEL_SEPARATOR = "\n"
SHOT_SEPARATOR = "\n\n"


dataset_config = {

    "trec": (PromptTemplate(
        preface="Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n",
        query_prefix="Question:",
        label_prefix="Answer Type:",
        prefix_sample_separator=PREFIX_SAMPLE_SEPARATOR,
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'Number', 1: 'Location', 2: 'Person', 3: 'Description', 4: 'Entity', 5: 'Abbreviation'}),
    
    "sst2": (PromptTemplate(
        preface="",
        query_prefix="Review:",
        label_prefix="Sentiment:",
        prefix_sample_separator=PREFIX_SAMPLE_SEPARATOR,
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'Negative', 1: 'Positive'}),
    
    "agnews": (PromptTemplate(
        preface="Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n",
        query_prefix="Article:",
        label_prefix="Answer:",
        prefix_sample_separator=PREFIX_SAMPLE_SEPARATOR,
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}),
    
    "dbpedia": (PromptTemplate(
        preface="Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n",
        query_prefix="Article:",
        label_prefix="Answer:",
        prefix_sample_separator=PREFIX_SAMPLE_SEPARATOR,
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'Company', 1: 'School', 2: 'Artist', 3: 'Athlete', 4: 'Politician', 5: 'Transportation', 6: 'Building', 7: 'Nature', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'Book'}),

}

def main():

    # Parse command line arguments and read config file
    print("Loading config...")
    args = parse_args()
    config = read_config(args.config_file)
    save_config(config, os.path.join(args.results_dir, "config.json"))

    # Instantiate base model
    print("Loading base model...")
    base_model, tokenizer = load_base_model(config["model"],checkpoints_dir=args.checkpoints_dir)

    # Iterate over datasets
    dataset_bar = tqdm(config["datasets"], desc="Dataset")
    for dataset_name in dataset_bar:
        dataset_bar.set_description(f"Dataset {ClassificationDatasetDict.short2name[dataset_name]}")
        if os.path.exists(os.path.join(args.results_dir,dataset_name)):
            continue

        # Instantiate classification model and dataset
        template, labels_dict = dataset_config[dataset_name]
        dataset = ClassificationDatasetDict(
            dataset_name,
            args.data_dir,
            n_shots=0,
            num_train_samples=config["num_train_samples"],
            num_test_samples=config["num_test_samples"],
            random_state=config["random_state"],
            sort_by_length=True,
            ascending=False
        )
        model = FewShotLanguageModelClassifier(
            base_model=base_model,
            tokenizer=tokenizer, 
            labels_dict=labels_dict
        )
        result = run(
            model,
            dataset,
            tokenizer=tokenizer,
            template=template,
            batch_size = config["batch_size"]
        )

        # Save results
        for key in result:
            dir_name = os.path.join(args.results_dir,dataset_name)
            os.makedirs(dir_name,exist_ok=True)
            np.save(os.path.join(dir_name,f"{key}.npy"),result[key])


def get_split_probs(
    model, 
    dataset,
    tokenizer,
    template,
    shots,
    batch_size = 32,
):
    loader = SequentialLoaderWithDataCollator(
        dataset=dataset,
        tokenizer=tokenizer,
        template=template,
        shots=shots,
        batch_size=batch_size
    )

    all_logprobs = []
    all_labels = []
    train_bar = tqdm(loader, desc="Batches", total=len(loader))
    for batch in train_bar:
        with torch.no_grad():
            _, logits = model(batch["sentences"])
            logprobs = torch.log_softmax(logits,dim=-1).cpu().numpy()
        all_logprobs.append(logprobs)
        all_labels.append(batch["labels"])
    all_logprobs = np.concatenate(all_logprobs,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    return all_logprobs, all_labels

def run(
    model, 
    dataset,
    tokenizer,
    template,
    batch_size = 32,
):
    # Train logprobs:
    train_logprobs, train_labels = get_split_probs(model, dataset["train"], tokenizer, template, dataset["shots"], batch_size)
    
    # Test logprobs:
    test_logprobs, test_labels = get_split_probs(model, dataset["test"], tokenizer, template, dataset["shots"], batch_size)

    results = {
        "train.logprobs": train_logprobs,
        "train.labels": train_labels,
        "test.logprobs": test_logprobs,
        "test.labels": test_labels
    }

    return results
            
        

if __name__ == "__main__":
    main()