
from accelerate import init_empty_weights, Accelerator
from ucpa.data import PromptTemplate
from ucpa.models import load_base_model
from ucpa.models.base import FewShotLanguageModelClassifier
import torch


queries_batch = [
    "I love this movie!",
    "I hate this movie!",
    "The movie was ok but the performance of the actors was terrible!",
    "The movie was ok but the performance of the actors was great!"
    # "This is not my favorite movie.",
    # "The movie was terrible"
]

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

def main():

    model_name = "t5-small"
    checkpoints_dir = "checkpoints"    

    # with init_empty_weights():
    base_model, tokenizer = load_base_model(model_name,checkpoints_dir)
    model = FewShotLanguageModelClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        labels_dict=labels_dict,
        template=template,
        sentences_shots=None,
        labels_shots=None
    )
    
    device = accelerator.device
    model.to(device)
    model = accelerator.prepare(model)
    model.eval()
    with torch.no_grad():
        _, logits = model(queries_batch)

    print(logits)

if __name__ == '__main__':
    main()