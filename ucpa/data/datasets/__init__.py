from .tony_zhao import (
    TonyZhaoTREC,
    TonyZhaoSST2,
    TonyZhaoAGNEWS,
    TonyZhaoDBPEDIA
)

dataset_name2class = {
    "tony_zhao_trec": TonyZhaoTREC,
    "tony_zhao_sst2": TonyZhaoSST2,
    "tony_zhao_agnews": TonyZhaoAGNEWS,
    "tony_zhao_dbpedia": TonyZhaoDBPEDIA
}
