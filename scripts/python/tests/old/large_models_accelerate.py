
from accelerate import init_empty_weights, Accelerator
from ucpa.data import PromptTemplate
from ucpa.models import load_base_model
from ucpa.models.base import FewShotLanguageModelClassifier
import torch
from torch.utils.data import DataLoader, Dataset


queries_batch = [
    "I love this movie!",
    "I hate this movie!",
    "The movie was ok but the performance of the actors was terrible!",
    "The movie was ok but the performance of the actors was great!"
    # "This is not my favorite movie.",
    # "The movie was terrible"
]
labels_batch = [1, 0, 0, 1]

template = PromptTemplate(
    preface="",
    query_prefix="Review:",
    label_prefix="Sentiment:",
    prefix_sample_separator=" ",
    query_label_separator="\n",
    shot_separator="\n\n"
)

labels_dict = {0: 'Negative', 1: 'Positive'}

accelerator = Accelerator()

class TestDataset(Dataset):

    def __init__(self, queries_batch, labels_batch):
        self.queries_batch = queries_batch
        self.labels_batch = labels_batch

    def __len__(self):
        return len(self.queries_batch)
    
    def __getitem__(self,idx):
        return self.queries_batch[idx], self.labels_batch[idx]


class DataCollator:

    def __init__(self, tokenizer, template, sentences_shots=None, labels_shots=None):
        self.tokenizer = tokenizer
        self.template = template
        self.sentences_shots = sentences_shots
        self.labels_shots = labels_shots

    def __call__(self, queries_batch, labels_batch):
        prompts_batch = [self.template.construct_prompt(query,sentences_shots=self.sentences_shots,labels_shots=self.labels_shots) for query in queries_batch]
        encoded_prompts = self.tokenizer(prompts_batch, return_tensors="pt", padding=True)
        return encoded_prompts, torch.tensor(labels_batch)


def main():

    model_name = "t5-small"
    # model_name = "gpt2"
    checkpoints_dir = "checkpoints"    

    # with init_empty_weights():
    base_model, tokenizer = load_base_model(model_name,checkpoints_dir)
    model = FewShotLanguageModelClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        labels_dict=labels_dict,
    )

    dataloader = DataLoader(
        dataset=TestDataset(queries_batch, labels_batch),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=DataCollator(
            tokenizer=tokenizer,
            template=template,
            sentences_shots=None,
            labels_shots=None
        )
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()

    device = accelerator.device
    model.to(device)
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    for queries, labels in dataloader:
        _, logits = model(queries)
        loss = loss_function(logits, labels)
        accelerator.backward(loss)
        optimizer.step()

if __name__ == '__main__':
    main()