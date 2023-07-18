

import os
import numpy as np
import torch
from ucpa.utils import parse_args, read_config, save_config
from ucpa.data import PromptTemplate, ClassificationDataset, BatchLoader
from ucpa.models.base import load_base_model, FewShotClassificationLanguageModel
from tqdm import tqdm
from copy import deepcopy

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
    model, tokenizer = load_base_model(config["model"],checkpoints_dir=args.checkpoints_dir)

    # Iterate over datasets
    dataset_bar = tqdm(config["datasets"], desc="Dataset")
    for dataset_name in dataset_bar:
        dataset_bar.set_description(f"Dataset {ClassificationDataset.short2name[dataset_name]}")
        if os.path.exists(os.path.join(args.results_dir,dataset_name)):
            continue

        # Instantiate classification model and dataset
        template, labels_dict = dataset_config[dataset_name]
        dataset = ClassificationDataset(
            dataset_name,
            args.data_dir,
            n_shots=0,
            num_train_samples=config["num_train_samples"],
            num_test_samples=config["num_test_samples"],
            random_state=config["random_state"]
        )
        classification_model = FewShotClassificationLanguageModel(
            model=model,
            tokenizer=tokenizer,
            template=template,
            labels_dict=labels_dict,
            sentences_shots=dataset["shots"]["sentences"],
            labels_shots=dataset["shots"]["labels"]
        )
        result = run(
            classification_model,
            dataset,
            batch_size = config["batch_size"]
        )

        # Save results
        for key in result:
            np.save(os.path.join(args.results_dir,dataset_name,f"{key}.npy"),result[key])

def sort_split_by_sentence_length(data):
    sorted_idx = np.argsort([len(sentence) for sentence in data["sentences"]])
    sorted_data = {}
    for key in data:
        sorted_data[key] = [data[key][idx] for idx in sorted_idx]
    sorted_data = deepcopy(sorted_data)
    return sorted_data


def get_split_probs(
    clf_model, 
    dataset,
    split,
    batch_size = 32,
):
    sorted_data = sort_split_by_sentence_length(dataset[split])
    loader = BatchLoader(sorted_data,batch_size)

    all_logprobs = []
    all_labels = []
    train_bar = tqdm(loader, desc=split, total=len(loader))
    for batch in train_bar:
        with torch.no_grad():
            logprobs = clf_model(batch["sentences"]).cpu().numpy()
        all_logprobs.append(logprobs)
        all_labels.append(batch["labels"])
    all_logprobs = np.concatenate(all_logprobs,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    return all_logprobs, all_labels

def run(
    clf_model, 
    dataset,
    batch_size = 32,
):
    # Train logprobs:
    train_logprobs, train_labels = get_split_probs(clf_model,dataset,"train",batch_size)
    
    # Test logprobs:
    test_logprobs, test_labels = get_split_probs(clf_model,dataset,"test",batch_size)

    results = {
        "train.logprobs": train_logprobs,
        "train.labels": train_labels,
        "test.logprobs": test_logprobs,
        "test.labels": test_labels
    }

    return results
            
        

if __name__ == "__main__":
    main()