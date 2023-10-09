from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
from .templates.utils import load_template



class BaseClassificationDatasetDict:
    """ Base class for classification datasets."""

    def __init__(self, data_dir, num_train_samples=None, num_test_samples=None, random_state=None, sort_by_length=True, ascending=False, template_args=None):
        """ Initialize dataset and load data. """
        
        data, self.labels_dict = self._load_data(data_dir)

        # Set random state
        self.random_state = random_state
        self._rs = np.random.RandomState(random_state)

        # Load template
        if template_args["name"] == "preface_plus_shots":
            # Remove shots from data
            template_args = deepcopy(template_args)
            n_shots = template_args.pop("n_shots")
            data, ids_shots, sentences_shots, labels_idx = self._get_random_shots(data, n_shots)
            labels_shots = [self.labels_dict[idx] for idx in labels_idx] if labels_idx is not None else None
            # Add shots to template_args dict
            template_args["shots"] = ShotsContainer(ids_shots, sentences_shots, labels_shots)
        self.template = load_template(**template_args)

        # Subsample data if needed
        data = self._shuffle_and_subsample(data, num_train_samples, num_test_samples)
        self.data = {
            split: ClassificationDataSplit(
                ids=data[f"{split}_ids"],
                sentences=data[f"{split}_sentences"],
                labels=data[f"{split}_labels"],
                template=self.template,
                sort_by_length=sort_by_length, 
                ascending=ascending
            ) for split in ["train", "test"]
        }


    def _get_random_shots(self, data, n_shots):

        if n_shots == 0:
            return data, None, None, None

        # Remove the prompt shots from the training set
        train_shots_idx = self._rs.permutation(len(data["train_sentences"]))[:n_shots]
        all_train_ids, all_train_sentences, all_train_labels = data["train_ids"], data['train_sentences'], data['train_labels']
        new_train_ids, new_train_sentences, new_train_labels = [], [], []
        for idx, (i, sentence, label) in enumerate(zip(all_train_ids, all_train_sentences, all_train_labels)):
            if idx not in train_shots_idx:
                new_train_ids.append(i)
                new_train_sentences.append(sentence)
                new_train_labels.append(label)

        # Create shots lists
        data['train_ids'], data['train_sentences'], data['train_labels'] = new_train_ids, new_train_sentences, new_train_labels
        ids_shots, sentences_shots, labels_shots = [], [], []
        for idx in train_shots_idx:
            ids_shots.append(all_train_ids[idx])
            sentences_shots.append(all_train_sentences[idx])
            labels_shots.append(all_train_labels[idx])
        
        return data, ids_shots, sentences_shots, labels_shots


    def _shuffle_and_subsample(self, data, num_train_samples=None, num_test_samples=None):
        """ Shuffle and subsample data. """
        if num_train_samples is not None:
            train_idx = self._rs.permutation(len(data["train_sentences"]))[:num_train_samples]
            data["train_ids"] = [data["train_ids"][idx] for idx in train_idx]
            data["train_sentences"] = [data["train_sentences"][idx] for idx in train_idx]
            data["train_labels"] = [data["train_labels"][idx] for idx in train_idx]
        if num_test_samples is not None:
            test_idx = self._rs.permutation(len(data["test_sentences"]))[:num_test_samples]
            data["test_ids"] = [data["test_ids"][idx] for idx in test_idx]
            data["test_sentences"] = [data["test_sentences"][idx] for idx in test_idx]
            data["test_labels"] = [data["test_labels"][idx] for idx in test_idx]

        return data
        
    def __getitem__(self, key):
        """ Return split of the dataset or shots container. """
        if key in ["train", "test"]:
            return self.data[key]
        elif key == "shots":
            return self.shots
        else:
            raise KeyError(f"{key} not in DataDict")
        
    @staticmethod
    def _load_data(data_dir):
        raise NotImplementedError
        

class ClassificationDataSplit(Dataset):

    def __init__(self, ids, sentences, labels, template, sort_by_length=True, ascending=False):
        if sort_by_length:
            sorted_idx = np.argsort([len(sentence) for sentence in sentences])
            if not ascending:
                sorted_idx = sorted_idx[::-1]
            sorted_ids, sorted_sentences, sorted_labels = [], [], []
            for idx in sorted_idx:
                sorted_ids.append(ids[idx])
                sorted_sentences.append(sentences[idx])
                sorted_labels.append(labels[idx])
        else:
            sorted_ids = ids
            sorted_sentences = sentences
            sorted_labels = labels
        
        self.ids = sorted_ids
        self.sentences = sorted_sentences
        self.labels = sorted_labels
        self.template = template

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            "id": self.ids[idx], 
            "sentence": self.sentences[idx], 
            "label": self.labels[idx]
        }
    

class ShotsContainer(dict):

    def __init__(self, ids, sentences, labels):
        self.ids = ids
        self.sentences_shots = sentences
        self.labels_shots = labels
        super().__init__(ids=ids, sentences=sentences, labels=labels)

    def __getitem__(self, key):
        if key in ["ids", "sentences", "labels"]:
            return super().__getitem__(key)
        else:
            raise ValueError("Key must be either 'ids', 'sentences' or 'labels'.")

    def __setitem__(self, key, value):
        raise ValueError("ShotsContainer is read-only.")

    def __repr__(self):
        return f"ShotsContainer(sentences={self.sentences_shots}, labels={self.labels_shots})"

    def __str__(self):
        return f"ShotsContainer(sentences={self.sentences_shots}, labels={self.labels_shots})"