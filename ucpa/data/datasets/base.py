import numpy as np
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    _split = None
    _splits = None

    def __init__(
        self, 
        data_dir, 
        split, 
        template, 
        num_samples=None, 
        sort_by_length=True, 
        ascending=False, 
        random_state=None
    ):

        # Validate split:
        if split not in self.splits:
            raise ValueError(f"Split must be one of {self.splits} but found {split}")
        
        # Set random state
        self.random_state = random_state
        self._rs = np.random.RandomState(random_state)

        # Load data, shuffle, subsample and sort by length:
        data, self.labels_dict = self._load_data(data_dir, split=split)
        data = self._shuffle_and_subsample(data, num_subsamples=num_samples)
        if sort_by_length:
            sorted_idx = np.argsort([len(sentence) for sentence in data["sentences"]])
            if not ascending:
                sorted_idx = sorted_idx[::-1]
            new_data = {key: [] for key in data}
            for idx in sorted_idx:
                for key in data:
                    new_data[key].append(data[key][idx])
        else:
            new_data = data
        
        self._data = new_data
        self._split = split
        self.template = template

    def _shuffle_and_subsample(self, data, num_subsamples=None):
        """ Shuffle and subsample data. """
        if num_subsamples is not None:
            idx = self._rs.permutation(len(data["sentences"]))[:num_subsamples]
        else:
            idx = self._rs.permutation(len(data["sentences"]))
        data["original_ids"] = [data["original_ids"][i] for i in idx]
        data["sentences"] = [data["sentences"][i] for i in idx]
        data["labels"] = [data["labels"][i] for i in idx]
        return data

    def __len__(self):
        return len(self._data["sentences"])
    
    def __getitem__(self, idx):
        return {
            "original_id": self._data["original_ids"][idx],
            "sentence": self._data["sentences"][idx], 
            "label": self._data["labels"][idx]
        }
    
    @property
    def split(self):
        if self._split not in ["train", "validation", "test"]:
            raise ValueError("Corrupted dataset")
        return self._split
    
    @property
    def splits(self):
        if not set(self._splits).issubset(["train", "validation", "test"]):
            raise ValueError("Corrupted data splits")
        return self._splits

