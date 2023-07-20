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
    labels_logprobs_base = torch.zeros((len(queries_batch),len(labels_dict)))
    for idx in range(len(labels_dict)):
        label = labels_dict[idx]
        tokenized_decoder_input = tokenizer([f"{tokenizer.pad_token} {label}"] * len(queries_batch),return_tensors="pt")
        with torch.no_grad():
            output_base = base_model(
                input_ids=tokenized_prompts_batch["input_ids"],
                attention_mask=tokenized_prompts_batch["attention_mask"],
                decoder_input_ids=tokenized_decoder_input["input_ids"],
                decoder_attention_mask=tokenized_decoder_input["attention_mask"]
            ).logits[:,:-1,:]
            logprobs_base = torch.log_softmax(output_base,dim=-1)
            labels_logprobs_base[:,idx] = torch.gather(
                logprobs_base,
                dim=-1,
                index=tokenized_decoder_input["input_ids"][:,1:].unsqueeze(-1)
            ).squeeze(-1).sum(dim=1)
    return labels_logprobs_base

def run_fewshot_model(model_name,checkpoints_dir):
    sentences_shots=None
    labels_shots=None
    base_model, tokenizer = load_base_model(model_name,checkpoints_dir)
    model = FewShotLanguageModelClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        labels_dict=labels_dict
    )
    model.eval()
    with torch.no_grad():
        prompts_batch = [template.construct_prompt(query,sentences_shots=sentences_shots,labels_shots=labels_shots) for query in queries_batch]
        encoded_prompts = tokenizer(prompts_batch, return_tensors="pt", padding=True)
        _, logits = model(encoded_prompts)
    return logits

def main():
    args = parse_args()
    model_name = "t5-small"
    labels_logprobs_base = run_base_model(model_name,args.checkpoints_dir)
    labels_logprobs_fewshot = run_fewshot_model(model_name,args.checkpoints_dir)
    print(labels_logprobs_base)
    print(labels_logprobs_fewshot)



if __name__ == '__main__':
    main()