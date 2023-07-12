
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PromptTemplate:
    """ Prompt template for a few-shot classification task."""

    def __init__(self,preface,query_prefix,label_prefix,query_label_separator="\n",shot_separator="\n\n"):
        self.preface = preface
        self.query_prefix = query_prefix
        self.label_prefix = label_prefix
        self.query_label_separator = query_label_separator
        self.shot_separator = shot_separator

    def construct_prompt(self,query,sentences_shots=None,labels_shots=None):
        if sentences_shots is None and labels_shots is None:
            shots_str = ""
        elif sentences_shots is not None and labels_shots is not None:
            if len(sentences_shots) != len(labels_shots):
                raise ValueError("Sentence and label shots must be the same length.")
            shots_str = self.shot_separator.join(f"{self.query_prefix}{s}{self.query_label_separator}{self.label_prefix}{l}" for s, l in zip(sentences_shots, labels_shots))
        else:
            raise ValueError("Sentence and label shots must either both be None or both be lists of strings.")
        prompt = f"{self.preface}{shots_str}{self.query_prefix}{query}{self.query_label_separator}{self.label_prefix}"
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



class ClassificationDataset(Dataset):
    """ Few-shot classification dataset."""

    def __init__(self,dataset_name,split="train",data_dir="",template=None,label_dict=None):
        """ Initialize dataset and load data. """
        self.dataset_name = dataset_name
        self.template = template
        self.label_dict = label_dict
        self.inv_label_dict = {v: k for k, v in label_dict.items()}
        self.split = split
        if dataset_name == "trec":
            self._data = load_trec(data_dir)
        elif dataset_name == "sst2":
            self._data = load_sst2(data_dir)
        elif dataset_name == "agnews":
            self._data = load_agnews(data_dir)
        elif dataset_name == "dbpedia":
            self._data = load_dbpedia(data_dir)

    def set_shots(self, sentences_shots=None, labels_shots=None, n_shots=None, random_state=None):
        """ Set which shots to use for the prompt and create the prompt constructor function. """
        if sentences_shots is not None and labels_shots is not None:
            if n_shots is not None or random_state is not None:
                raise ValueError("Cannot specify both sentence and label shots and n_shots or random_state.")
            self.construct_prompt = lambda query: self.template.construct_prompt(query,sentences_shots,labels_shots)
        elif (sentences_shots is None and labels_shots is not None) or (sentences_shots is not None and labels_shots is None):
            raise ValueError("Sentence shots cannot be None if label shots are not None and viceversa.")
        elif n_shots is not None and random_state is not None:
            self.construct_prompt = lambda query: self.template.construct_prompt(query,*self._get_random_shots(n_shots,random_state))
        elif n_shots is None:
            raise ValueError("Must specify either sentence and label shots or n_shots.")
        
    def _get_random_shots(self,n_shots,random_state=None):

        if n_shots == 0:
            return [], []
        
        # Remove the prompt shots from the training set
        rs = np.random.RandomState(random_state)
        train_shots_idx = rs.permutation(len(self._data["train_sentences"]))[:n_shots]
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
        
    def __len__(self):
        """ Return number of examples in dataset. """
        if self.split == "train":
            return len(self._data["train_sentences"])
        elif self.split == "test":
            return len(self._data["test_sentences"])
        else:
            raise ValueError("Split must be either 'train' or 'test'.")
        
    def __getitem__(self, idx):
        """ Return example at index idx. """
        if self.split == "train":
            return self.construct_prompt(self._data["train_sentences"][idx]), self._data["train_labels"][idx]
        elif self.split == "test":
            return self.construct_prompt(self._data["test_sentences"][idx]), self._data["test_labels"][idx]
        else:
            raise ValueError("Split must be either 'train' or 'test'.")