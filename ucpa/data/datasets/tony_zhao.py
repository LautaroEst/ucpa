from ..base import BaseClassificationDatasetDict
import pandas as pd
import numpy as np


class TonyZhaoTREC(BaseClassificationDatasetDict):

    @staticmethod
    def _load_data(data_dir):
        labels_dict = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Abbreviation']}
        inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
        
        train_sentences = []
        train_labels = []
        with open(f'{data_dir}/tony_zhao/trec/train.txt', 'r') as train_data:
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
        with open(f'{data_dir}/tony_zhao/trec/test.txt', 'r') as test_data:
            for line in test_data:
                test_label = line.split(' ')[0].split(':')[0]
                test_label = inv_label_dict[test_label]
                test_sentence = ' '.join(line.split(' ')[1:]).strip()
                test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
                test_labels.append(test_label)
                test_sentences.append(test_sentence)
        
        data = {
            'train_ids': list(range(len(train_sentences))),
            'train_sentences': train_sentences,
            'train_labels': train_labels,
            'test_ids': list(range(len(test_sentences))),
            'test_sentences': test_sentences,
            'test_labels': test_labels
        }
        return data, labels_dict


class TonyZhaoSST2(BaseClassificationDatasetDict):

    @staticmethod
    def _load_data(data_dir):
        labels_dict = {0: ['Negative'], 1: ['Positive']}

        def process_raw_data_sst(lines):
            """from lines in dataset to two lists of sentences and labels respectively"""
            labels = []
            sentences = []
            for line in lines:
                labels.append(int(line[0]))
                sentences.append(line[2:].strip())
            return sentences, labels

        with open(f"{data_dir}/tony_zhao/sst2/stsa.binary.train", "r") as f:
            train_lines = f.readlines()
        with open(f"{data_dir}/tony_zhao/sst2/stsa.binary.test", "r") as f:
            test_lines = f.readlines()
        train_sentences, train_labels = process_raw_data_sst(train_lines)
        test_sentences, test_labels = process_raw_data_sst(test_lines)

        data = {
            'train_ids': list(range(len(train_sentences))),
            'train_sentences': train_sentences,
            'train_labels': train_labels,
            'test_ids': list(range(len(test_sentences))),
            'test_sentences': test_sentences,
            'test_labels': test_labels
        }
        return data, labels_dict
    

class TonyZhaoAGNEWS(BaseClassificationDatasetDict):

    @staticmethod
    def _load_data(data_dir):

        labels_dict = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology']}

        train_data = pd.read_csv(f'{data_dir}/tony_zhao/agnews/train.csv')
        test_data = pd.read_csv(f'{data_dir}/tony_zhao/agnews/test.csv')

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

        ##########################################################
        rs = np.random.RandomState(0)
        test_idx = rs.permutation(len(test_sentences))[:1000]
        test_sentences = [test_sentences[i] for i in test_idx]
        test_labels = [test_labels[i] for i in test_idx]
        ##########################################################

        data = {
            'train_ids': list(range(len(train_sentences))),
            'train_sentences': train_sentences,
            'train_labels': train_labels,
            'test_ids': list(range(len(test_sentences))),
            'test_sentences': test_sentences,
            'test_labels': test_labels
        }
        return data, labels_dict
    

class TonyZhaoDBPEDIA(BaseClassificationDatasetDict):

    @staticmethod
    def _load_data(data_dir):

        labels_dict = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Athlete'], 4: ['Politician'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}

        train_data = pd.read_csv(f'{data_dir}/tony_zhao/dbpedia/train_subset.csv')
        test_data = pd.read_csv(f'{data_dir}/tony_zhao/dbpedia/test.csv')

        train_sentences = train_data['Text']
        train_sentences = list([item.replace('""', '"') for item in train_sentences])
        train_labels = list(train_data['Class'])

        test_sentences = test_data['Text']
        test_sentences = list([item.replace('""', '"') for item in test_sentences])
        test_labels = list(test_data['Class'])
        
        train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
        test_labels = [l - 1 for l in test_labels]

        ##########################################################
        rs = np.random.RandomState(1)
        test_idx = rs.permutation(len(test_sentences))[:1000]
        test_sentences = [test_sentences[i] for i in test_idx]
        test_labels = [test_labels[i] for i in test_idx]
        ##########################################################

        data = {
            'train_ids': list(range(len(train_sentences))),
            'train_sentences': train_sentences,
            'train_labels': train_labels,
            'test_ids': list(range(len(test_sentences))),
            'test_sentences': test_sentences,
            'test_labels': test_labels
        }
        return data, labels_dict