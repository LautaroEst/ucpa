from ..datasets import dataset_name2class

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
        raise ValueError("ShotsContainer is read-only.")

    def __repr__(self):
        return f"ShotsContainer(sentences={self.sentences_shots}, labels={self.labels_shots})"

    def __str__(self):
        return f"ShotsContainer(sentences={self.sentences_shots}, labels={self.labels_shots})"
    
    @classmethod
    def from_dataset(cls, data_dir, dataset, split, n_shots, random_state=None):
        dataset_cls = dataset_name2class[dataset]
        dataset = dataset_cls(
            data_dir=data_dir,
            split=split, 
            template=None,
            num_samples=n_shots,
            sort_by_length=False, 
            ascending=False, 
            random_state=random_state
        )
        sentences = [dataset[i]["sentence"] for i in range(n_shots)]
        labels = [dataset.labels_dict[dataset[i]["label"]] for i in range(n_shots)]
        return cls(sentences, labels)