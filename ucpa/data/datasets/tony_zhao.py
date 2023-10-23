from .base import ClassificationDataset
import pandas as pd
import numpy as np


class TonyZhaoTREC(ClassificationDataset):

    _splits = ["train", "test"]

    @staticmethod
    def _load_data(data_dir, split="train"):
        labels_dict = {0: 'Number', 1: 'Location', 2: 'Person', 3: 'Description', 4: 'Entity', 5: 'Abbreviation'}
        inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
        
        sentences = []
        labels = []
        with open(f'{data_dir}/tony_zhao/trec/{split}.txt', 'r') as data:
            for line in data:
                label = line.split(' ')[0].split(':')[0]
                label = inv_label_dict[label]
                sentence = ' '.join(line.split(' ')[1:]).strip()
                # basic cleaning
                sentence = sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
                labels.append(label)
                sentences.append(sentence)

        data = {
            'original_ids': list(range(len(sentences))),
            'sentences': sentences,
            'labels': labels,
        }
        return data, labels_dict


class TonyZhaoSST2(ClassificationDataset):

    _splits = ["train", "test"]

    @staticmethod
    def _load_data(data_dir, split="train"):
        labels_dict = {0: 'Negative', 1: 'Positive'}
            
        # from lines in dataset to two lists of sentences and labels respectively
        with open(f"{data_dir}/tony_zhao/sst2/stsa.binary.{split}", "r") as f:
            lines = f.readlines()
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())

        data = {
            'original_ids': list(range(len(sentences))),
            'sentences': sentences,
            'labels': labels
        }
        return data, labels_dict
    

class TonyZhaoAGNEWS(ClassificationDataset):

    _splits = ["train", "test"]

    @staticmethod
    def _load_data(data_dir, split="train"):
        labels_dict = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}

        data = pd.read_csv(f'{data_dir}/tony_zhao/agnews/{split}.csv')
        sentences = data['Title'] + ". " + data['Description']
        sentences = list(
            [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item in sentences]
        ) # some basic cleaning
        labels = [l - 1 for l in list(data['Class Index'])] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4

        ##########################################################
        if split == "test":
            rs = np.random.RandomState(0)
            idx = rs.permutation(len(sentences))[:1000]
            sentences = [sentences[i] for i in idx]
            labels = [labels[i] for i in idx]
        ##########################################################

        data = {
            'original_ids': list(range(len(sentences))),
            'sentences': sentences,
            'labels': labels,
        }
        return data, labels_dict
    

class TonyZhaoDBPEDIA(ClassificationDataset):

    _splits = ["train", "test"]

    @staticmethod
    def _load_data(data_dir, split="train"):
        labels_dict = {0: 'Company', 1: 'School', 2: 'Artist', 3: 'Athlete', 4: 'Politician', 5: 'Transportation', 6: 'Building', 7: 'Nature', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'Book'}

        if split == "train":
            data = pd.read_csv(f'{data_dir}/tony_zhao/dbpedia/train_subset.csv')
        else:
            data = pd.read_csv(f'{data_dir}/tony_zhao/dbpedia/test.csv')

        sentences = data['Text']
        sentences = list([item.replace('""', '"') for item in sentences])
        labels = [l - 1 for l in list(data['Class'])] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...

        ##########################################################
        if split == "test":
            rs = np.random.RandomState(1)
            idx = rs.permutation(len(sentences))[:1000]
            sentences = [sentences[i] for i in idx]
            labels = [labels[i] for i in idx]
        ##########################################################

        data = {
            'original_ids': list(range(len(sentences))),
            'sentences': sentences,
            'labels': labels
        }
        return data, labels_dict