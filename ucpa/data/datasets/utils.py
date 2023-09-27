from . import *

dataset_name2class = {
    "tony_zhao_trec": TonyZhaoTREC,
    "tony_zhao_sst2": TonyZhaoSST2,
    "tony_zhao_agnews": TonyZhaoAGNEWS,
    "tony_zhao_dbpedia": TonyZhaoDBPEDIA
}


def load_dataset(name, *args, **kwargs):
    dataset_cls = dataset_name2class[name]
    return dataset_cls(*args, **kwargs)