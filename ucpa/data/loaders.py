
import torch
from torch.utils.data import DataLoader

class LMClassificationDataCollator:

    def __init__(self, tokenizer, template, labels):
        self.tokenizer = tokenizer
        self.template = template
        self.labels = labels
        num_of_prompt_tokens_without_query = len(tokenizer.tokenize(template.construct_prompt(" ")))
        self.max_query_tokens = tokenizer.max_len_single_sentence - num_of_prompt_tokens_without_query - max([len(tokenizer.tokenize(l)) for l in labels]) - 2

    def __call__(self, batch):

        ids, prompts, labels = [], [], []
        for sample in batch:
            ids.append(sample["original_id"])
            prompts.append(self.template.construct_prompt(self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(sample["sentence"])[:self.max_query_tokens])))
            labels.append(sample["label"])
        encoded_labels = {idx: {k: v.repeat(len(prompts),1) for k, v in self.tokenizer([f"{self.template.prefix_sample_separator}{label}"], return_tensors="pt", padding=True).items()} for idx, label in enumerate(self.labels)}

        return {
            "original_id": ids,
            "prompt": prompts,
            "encoded_prompt": self.tokenizer(prompts, return_tensors="pt", padding=True),
            "label": torch.tensor(labels),
            "encoded_labels": encoded_labels
        }

class SequentialLoaderWithDataCollator(DataLoader):

    def __init__(self, dataset, tokenizer, labels, batch_size=32, **kwargs):
        self.tokenizer = tokenizer
        self.labels = labels
        kwargs.pop("collate_fn", None)
        kwargs.pop("shuffle", None)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=LMClassificationDataCollator(
                tokenizer=tokenizer,
                template=dataset.template,
                labels=labels
            ),
            **kwargs
        )

class LMDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_query_tokens = tokenizer.max_len_single_sentence

    def __call__(self, batch):

        ids, prompts, srcs, tgts = [], [], [], []
        for sample in batch:
            ids.append(sample["text_id"])
            prompts.append(sample["original_text"])
            srcs.append(sample["src_text"])
            tgts.append(sample["tgt_text"])

        return {
            "original_id": ids,
            "prompt": prompts,
            "encoded_src_prompt": self.tokenizer(srcs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_query_tokens),
            "encoded_tgt_prompt": self.tokenizer(tgts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_query_tokens),
        }

class LanguageModelTextLoader(DataLoader):

    def __init__(self, dataset, tokenizer, batch_size=32, **kwargs):
        self.tokenizer = tokenizer
        kwargs.pop("collate_fn", None)
        kwargs.pop("shuffle", None)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=LMDataCollator(
                tokenizer=tokenizer,
            ),
            **kwargs
        )

