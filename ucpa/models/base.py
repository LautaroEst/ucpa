
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, List, Dict
from ..data import PromptTemplate


SUPPORTED_MODELS = {
    "decoder_only": [
        "gpt2",
        # "gpt2-medium",
        # "gpt2-large",
        # "gpt2-xl"
    ],
    "encoder_decoder": [
        "t5-small",
        # "t5-base",
        # "t5-large",
        # "t5-3b",
        # "t5-11b",
        # "google/flan-t5-xxl",
        # "google/flan-t5-base",
        # "google/flan-t5-xl",
        # "google/flan-t5-large",
        "google/flan-t5-small"
    ]
}


class LabelsDecoder(nn.Module):

    def __init__(
        self, 
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, 
        labels_dict: Dict[int,str]
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.labels_dict = labels_dict
        self.encoded_labels = self._encode_labels()
        self._forward = self._encoder_decoder_forward if self.base_model.config.is_encoder_decoder else self._decoder_only_forward

    def _encode_labels(self):
        return {idx: self.tokenizer([f" {label}"], return_tensors="pt", padding=True) for idx, label in self.labels_dict.items()}

    def _decoder_only_forward(self, encoder_output, decoder_input):
        batch_size = encoder_output["input_ids"].shape[0]
        label_len = decoder_input["attention_mask"].shape[1]
        sequence_lens = encoder_output["attention_mask"].sum(dim=-1,keepdim=True)
        logits = self.base_model(
            input_ids=decoder_input["input_ids"],
            attention_mask=torch.cat(
                (encoder_output["attention_mask"],torch.ones((batch_size,label_len),dtype=torch.long)),
                dim=1
            ),
            position_ids=torch.arange(label_len).repeat(batch_size,1) + sequence_lens,
            past_key_values=encoder_output["past_key_values"],
            output_attentions=False,
            output_hidden_states=False
        ).logits
        logits = torch.cat((encoder_output["logits"][:,-1,:].unsqueeze(1),logits[:,:-1,:]),dim=1)
        return logits
    
    def _encoder_decoder_forward(self, encoder_output, decoder_input):
        raise NotImplementedError(f"Architecture type encoder_decoder not implemented.")

    def forward(self, encoder_output):
        batch_size = encoder_output["input_ids"].shape[0]
        for idx in range(len(self.encoded_labels)):
            encoded_label = self.encoded_labels[idx]
            encoded_label = {k: v.repeat(batch_size,1) for k, v in encoded_label.items()}
            logits = self._forward(encoder_output, encoded_label)
            logprobs = torch.log_softmax(logits,dim=-1)
            gathered_logprobs = torch.gather(
                logprobs,
                dim=-1,
                index=encoded_label["input_ids"].unsqueeze(-1)
            ).squeeze(-1).sum(dim=1, keepdim=True)
            labels_logprobs.append(gathered_logprobs[:, 0])
        labels_logprobs = torch.stack(labels_logprobs,dim=1)
        return labels_logprobs


class PromptEncoder(nn.Module):

    def __init__(
        self, 
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, 
        template: PromptTemplate, 
        sentences_shots: Optional[List[str]] = None,
        labels_shots: Optional[List[str]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.template = template
        self.sentences_shots = sentences_shots
        self.labels_shots = labels_shots
        self._forward = self._encoder_decoder_forward if self.base_model.config.is_encoder_decoder else self._decoder_only_forward

    def _decoder_only_forward(self, encoded_prompts):
        position_ids = self.create_position_ids(encoded_prompts["attention_mask"])
        prompt_output = self.model(
            input_ids=encoded_prompts["input_ids"],
            attention_mask=encoded_prompts["attention_mask"],
            position_ids=position_ids,
            use_cache=True, 
            output_attentions=False, 
            output_hidden_states=False
        )
        encoder_output = {
            "input_ids": encoded_prompts["input_ids"],
            "attention_mask": encoded_prompts["attention_mask"],
            "past_key_values": prompt_output.past_key_values,
            "logits": prompt_output.logits
        }
        return encoder_output
    
    def _encoder_decoder_forward(self, encoded_prompt):
        raise NotImplementedError(f"Architecture type encoder_decoder not implemented.")

    def forward(self, queries_batch):
        prompts_batch = [self.template.construct_prompt(query,sentences_shots=self.sentences_shots,labels_shots=self.labels_shots) for query in queries_batch]
        encoded_prompts = self.tokenizer(prompts_batch, return_tensors="pt", padding=True)
        encoder_output = self._forward(encoded_prompts)
        return encoder_output

    @staticmethod
    def create_position_ids(attention_mask):
        position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        return position_ids



class FewShotLanguageModelClassifier(nn.Module):

    def __init__(
        self, 
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, 
        labels_dict: Dict[int,str],
        template: PromptTemplate, 
        sentences_shots: Optional[List[str]] = None,
        labels_shots: Optional[List[str]] = None
    ):
        super().__init__()
        self.prompt_encoder = PromptEncoder(base_model, tokenizer, template, sentences_shots, labels_shots)
        self.decoder = LabelsDecoder(base_model, tokenizer, labels_dict)

    def forward(self, batch_queries):
        encoder_output = self.prompt_encoder(batch_queries)
        labels_logprobs = self.labels_decoder(encoder_output)
        return labels_logprobs
        

    
    
    # def _encoder_decoder_forward(self,queries_batch):
    #     raise NotImplementedError(f"Architecture type encoder_decoder not implemented.")

    # def _decoder_only_forward(self,encoded_prompts):
    #     batch_size = encoded_prompts["input_ids"].shape[0]
    #     ## TODO: review this part ##
    #     encoded_prompts = {k: v.to(self.model.device) for k, v in encoded_prompts.items()}
    #     prompt_output = self.model(**encoded_prompts, use_cache=True, output_attentions=False, output_hidden_states=False)
    #     ############################
    #     last_token_logprobs = torch.log_softmax(prompt_output.logits[:,-1,:], dim=-1)
    #     sequence_lens = encoded_prompts["attention_mask"].sum(dim=-1,keepdim=True).cpu()
    #     labels_logprobs = []
    #     for idx in range(len(self.labels_dict)):
    #         label = self.labels_dict[idx]
    #         encoded_label = self.tokenizer([f" {label}" for _ in range(batch_size)], return_tensors="pt", padding=True)
    #         label_len = encoded_label["attention_mask"].shape[1]
    #         encoded_label["position_ids"] = torch.arange(label_len).repeat(batch_size,1) + sequence_lens
    #         encoded_label["attention_mask"] = torch.cat((encoded_prompts["attention_mask"].cpu(),torch.ones((batch_size,label_len),dtype=torch.long)),dim=1)
    #         encoded_label = {k: v.to(self.model.device) for k, v in encoded_label.items()}
    #         logprobs = torch.log_softmax(self.model(**encoded_label, past_key_values=prompt_output.past_key_values, output_attentions=False, output_hidden_states=False).logits,dim=-1)
    #         gathered_logprobs = torch.gather(
    #             logprobs[:,:-1,:],
    #             dim=-1,
    #             index=encoded_label["input_ids"][:, 1:].unsqueeze(-1)
    #         ).squeeze(-1).sum(dim=1,keepdim=True) + torch.gather(last_token_logprobs,dim=-1,index=encoded_label["input_ids"][:,-1].unsqueeze(-1))
    #         labels_logprobs.append(gathered_logprobs[:, 0])
    #     labels_logprobs = torch.stack(labels_logprobs,dim=1)
    #     return labels_logprobs



