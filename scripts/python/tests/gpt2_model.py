from utils import parse_args
from ucpa.models import load_base_model, FewShotLanguageModelClassifier
from ucpa.models.base import PromptEncoder
from ucpa.data import PromptTemplate
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


def run_base_model(model_name,checkpoints_dir):
    base_model, tokenizer = load_base_model(model_name,checkpoints_dir)
    base_model.eval()
    prompts_batch = []
    for query in queries_batch:
        prompts_batch.append(template.construct_prompt(query))
    tokenized_prompts_batch = tokenizer(
        prompts_batch,
        padding=True,
        return_tensors="pt"
    )
    tokenized_prompts_batch["position_ids"] = PromptEncoder.create_position_ids(tokenized_prompts_batch["attention_mask"])
    with torch.no_grad():
        output_base = base_model(**tokenized_prompts_batch).logits
    logprobs_base = torch.log_softmax(output_base,dim=-1)
    labels_logprobs_base = torch.zeros((len(queries_batch),len(labels_dict)))
    for idx in range(len(labels_dict)):
        label = labels_dict[idx]
        label_idx = tokenizer(f" {label}",return_tensors="pt")["input_ids"].item()
        labels_logprobs_base[:,idx] = logprobs_base[:,-1,label_idx].detach()
    return labels_logprobs_base

def run_fewshot_model(model_name,checkpoints_dir):
    base_model, tokenizer = load_base_model(model_name,checkpoints_dir)
    model = FewShotLanguageModelClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        labels_dict=labels_dict,
        template=template,
        sentences_shots=None,
        labels_shots=None
    )
    model.eval()
    with torch.no_grad():
        output = model(queries_batch)
    return output

def main():
    args = parse_args()
    labels_logprobs_base = run_base_model("gpt2",args.checkpoints_dir)
    labels_logprobs_fewshot = run_fewshot_model("gpt2",args.checkpoints_dir)
    print(labels_logprobs_base)
    print(labels_logprobs_fewshot)

        
    




if __name__ == '__main__':
    main()