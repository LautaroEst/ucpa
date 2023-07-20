
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PromptTemplate:
    """ Prompt template for a few-shot classification task."""

    def __init__(self,preface,query_prefix,label_prefix,prefix_sample_separator=" ",query_label_separator="\n",shot_separator="\n\n"):
        self.preface = preface
        self.query_prefix = query_prefix
        self.label_prefix = label_prefix
        self.prefix_sample_separator = prefix_sample_separator
        self.query_label_separator = query_label_separator
        self.shot_separator = shot_separator

    def construct_prompt(self,query,sentences_shots=None,labels_shots=None):
        if sentences_shots is None and labels_shots is None:
            shots_str = ""
        elif sentences_shots is not None and labels_shots is not None:
            if len(sentences_shots) != len(labels_shots):
                raise ValueError("Sentence and label shots must be the same length.")
            shots_str = self.shot_separator.join(f"{self.query_prefix}{self.prefix_sample_separator}{s}{self.query_label_separator}{self.label_prefix}{self.prefix_sample_separator}{l}" for s, l in zip(sentences_shots, labels_shots))
        else:
            raise ValueError("Sentence and label shots must either both be None or both be lists of strings.")
        prompt = f"{self.preface}{shots_str}{self.query_prefix}{self.prefix_sample_separator}{query}{self.query_label_separator}{self.label_prefix}"
        return prompt
            
        

def load_trec(data_dir):
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{data_dir}/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{data_dir}/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    
    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def load_sst2(data_dir):
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{data_dir}/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{data_dir}/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)

    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def load_agnews(data_dir):
    train_data = pd.read_csv(f'{data_dir}/agnews/train.csv')
    test_data = pd.read_csv(f'{data_dir}/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]

    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def load_dbpedia(data_dir):
    train_data = pd.read_csv(f'{data_dir}/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'{data_dir}/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]

    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data



class ClassificationDatasetDict:
    """ Few-shot classification dataset."""

    short2name = {
        "trec": "TREC",
        "sst2": "SST-2",
        "agnews": "AGNews",
        "dbpedia": "DBPedia"
    }

    def __init__(self,dataset_name,data_dir="",n_shots=0,num_train_samples=None,num_test_samples=None,random_state=None,sort_by_length=True,ascending=False):
        """ Initialize dataset and load data. """
        self.dataset_name = dataset_name
        if dataset_name == "trec":
            self._data = load_trec(data_dir)
        elif dataset_name == "sst2":
            self._data = load_sst2(data_dir)
        elif dataset_name == "agnews":
            self._data = load_agnews(data_dir)
        elif dataset_name == "dbpedia":
            self._data = load_dbpedia(data_dir)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}.")
        self.n_shots = n_shots
        self.random_state = random_state
        self._rs = np.random.RandomState(random_state)
        self.sentences_shots, labels_idx = self._get_random_shots(n_shots)
        self.labels_shots = [self.label_dict[idx] for idx in labels_idx] if labels_idx is not None else None
        self._shuffle_and_subsample(num_train_samples,num_test_samples)
        self.sort_by_length = sort_by_length
        self.ascending = ascending


    def _shuffle_and_subsample(self,num_train_samples,num_test_samples):
        """ Shuffle and subsample data. """
        if num_train_samples is not None:
            train_idx = self._rs.permutation(len(self._data["train_sentences"]))[:num_train_samples]
            self._data["train_sentences"] = [self._data["train_sentences"][idx] for idx in train_idx]
            self._data["train_labels"] = [self._data["train_labels"][idx] for idx in train_idx]
        if num_test_samples is not None:
            test_idx = self._rs.permutation(len(self._data["test_sentences"]))[:num_test_samples]
            self._data["test_sentences"] = [self._data["test_sentences"][idx] for idx in test_idx]
            self._data["test_labels"] = [self._data["test_labels"][idx] for idx in test_idx]
        if self.sentences_shots is not None and self.labels_shots is not None:
            shots_idx = self._rs.permutation(len(self.sentences_shots))
            self.sentences_shots = [self.sentences_shots[idx] for idx in shots_idx]
            self.labels_shots = [self.labels_shots[idx] for idx in shots_idx]
        
        
    def _get_random_shots(self,n_shots):

        if n_shots == 0:
            return None, None
        
        # Remove the prompt shots from the training set
        train_shots_idx = self._rs.permutation(len(self._data["train_sentences"]))[:n_shots]
        all_train_sentences = self._data['train_sentences']
        all_train_labels = self._data['train_labels']
        new_train_sentences = []
        new_train_labels = []
        for idx, (sentence, label) in enumerate(zip(all_train_sentences, all_train_labels)):
            if idx not in train_shots_idx:
                new_train_sentences.append(sentence)
                new_train_labels.append(label)
        self._data['train_sentences'] = new_train_sentences
        self._data['train_labels'] = new_train_labels
        sentences_shots = [all_train_sentences[idx] for idx in train_shots_idx]
        labels_shots = [all_train_labels[idx] for idx in train_shots_idx]
        return sentences_shots, labels_shots
        
    def __getitem__(self, split):
        """ Return split of the dataset. """
        if split in ["train", "test"]:
            return ClassificationDataset(
                sentences=self._data[f"{split}_sentences"], 
                labels=self._data[f"{split}_labels"],
                sort_by_length=self.sort_by_length, 
                ascending=self.ascending
            )
        elif split == "shots":
            return ShotsContainer(
                sentences=self.sentences_shots, 
                labels=self.labels_shots
            )
        else:
            raise ValueError("Split must be either 'train' or 'test'.")
        

class ShotsContainer(dict):

    def __init__(self, sentences, labels):
        self.sentences_shots = sentences
        self.labels_shots = labels
        super().__init__(sentences=sentences, labels=labels)

    def __getitem__(self, key):
        if key in ["sentences", "labels"]:
            return super().__getitem__(key)
        else:
            raise ValueError("Key must be either 'sentences' or 'labels'.")

    def __setitem__(self, key, value):
        if key in ["sentences", "labels"]:
            super().__setitem__(key, value)
        else:
            raise ValueError("Key must be either 'sentences' or 'labels'.")

    def __repr__(self):
        return f"ShotsContainer(sentences={self.sentences_shots}, labels={self.labels_shots})"

    def __str__(self):
        return f"ShotsContainer(sentences={self.sentences_shots}, labels={self.labels_shots})"


class ClassificationDataset(Dataset):

    def __init__(self, sentences, labels, sort_by_length=True, ascending=False):
        if sort_by_length:
            sorted_idx = np.argsort([len(sentence) for sentence in sentences])
            if not ascending:
                sorted_idx = sorted_idx[::-1]
            sorted_sentences = []
            sorted_labels = []
            for idx in sorted_idx:
                sorted_sentences.append(sentences[idx])
                sorted_labels.append(labels[idx])
        else:
            sorted_sentences = sentences
            sorted_labels = labels
        self.sentences = sorted_sentences
        self.labels = sorted_labels

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class DataCollator:

    def __init__(self, tokenizer, template, sentences_shots=None, labels_shots=None):
        self.tokenizer = tokenizer
        self.template = template
        self.sentences_shots = sentences_shots
        self.labels_shots = labels_shots

    def __call__(self, batch):
        queries_batch, labels_batch = zip(*batch)
        prompts_batch = [self.template.construct_prompt(query,sentences_shots=self.sentences_shots,labels_shots=self.labels_shots) for query in queries_batch]
        encoded_prompts = self.tokenizer(prompts_batch, return_tensors="pt", padding=True)
        return {
            "sentences": encoded_prompts,
            "labels": torch.tensor(labels_batch)
        }

class SequentialLoaderWithDataCollator(DataLoader):

    def __init__(self, dataset, tokenizer, template, shots, batch_size=32, **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollator(
                tokenizer=tokenizer,
                template=template,
                sentences_shots=shots["sentences"],
                labels_shots=shots["labels"]
            ),
        )
 