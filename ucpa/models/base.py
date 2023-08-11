
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, List, Dict
from ..data import PromptTemplate
import lightning.pytorch as pl


SUPPORTED_MODELS = {
    "decoder_only": [
        "gpt2",
        # "gpt2-medium",
        # "gpt2-large",
        "gpt2-xl",
        "meta-llama/Llama-2-7b-hf"
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
        if self.base_model.name_or_path in SUPPORTED_MODELS["decoder_only"]:
            self._forward = self._decoder_only_forward
        elif self.base_model.name_or_path in SUPPORTED_MODELS["encoder_decoder"]:
            self._forward = self._encoder_decoder_forward
        else:
            raise ValueError(f"Architecture type of {self.base_model.name_or_path} model not supported.")
        
    def _encode_labels(self):
        return {idx: self.tokenizer([f" {label}"], return_tensors="pt", padding=True) for idx, label in self.labels_dict.items()}

    def _decoder_only_forward(self, encoder_output, decoder_input):
        batch_size = encoder_output["input_ids"].shape[0]
        label_len = decoder_input["attention_mask"].shape[1]
        sequence_lens = encoder_output["attention_mask"].sum(dim=-1,keepdim=True)
        logits = self.base_model(
            input_ids=decoder_input["input_ids"],
            attention_mask=torch.cat(
                (
                    encoder_output["attention_mask"],
                    torch.ones(
                        (batch_size, label_len),
                        dtype=torch.long,
                        device=encoder_output["attention_mask"].device
                    )
                ),
                dim=1
            ),
            position_ids=torch.arange(label_len, device=sequence_lens.device).repeat(batch_size,1) + sequence_lens,
            past_key_values=encoder_output["past_key_values"],
            output_attentions=False,
            output_hidden_states=False
        ).logits
        logits = torch.cat((encoder_output["logits"][:,-1,:].unsqueeze(1),logits[:,:-1,:]),dim=1)
        return logits
    
    def _encoder_decoder_forward(self, encoder_output, decoder_input):
        pad_token_id = self.tokenizer.pad_token_id
        pad_tensor = torch.ones(
            decoder_input["input_ids"].shape[0], 1,
            dtype=decoder_input["input_ids"].dtype,
            device=decoder_input["input_ids"].device
        ) * pad_token_id
        decoder_input_ids = torch.cat((pad_tensor,decoder_input["input_ids"]),dim=1)
        
        pad_tensor = torch.ones(
            decoder_input["attention_mask"].shape[0], 1,
            dtype=decoder_input["attention_mask"].dtype,
            device=decoder_input["attention_mask"].device
        )
        decoder_attention_mask = torch.cat((pad_tensor,decoder_input["attention_mask"]),dim=1)
        out = self.base_model(
            attention_mask=encoder_output["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=(
                encoder_output["encoder_last_hidden_state"],
                encoder_output["encoder_hidden_states"],
                encoder_output["encoder_attentions"]
            )
        )
        logits = out.logits[:,:-1,:]
        return logits

    def forward(self, encoder_output):
        batch_size = encoder_output["input_ids"].shape[0]
        labels_logits = []
        for idx in range(len(self.encoded_labels)):
            encoded_label = self.encoded_labels[idx]
            encoded_label = {k: v.repeat(batch_size,1).to(device=encoder_output["input_ids"].device) for k, v in encoded_label.items()}
            logits = self._forward(encoder_output, encoded_label)
            logprobs = torch.log_softmax(logits,dim=-1)
            gathered_logits = torch.gather(
                logprobs,
                dim=-1,
                index=encoded_label["input_ids"].unsqueeze(-1)
            ).squeeze(-1).sum(dim=1, keepdim=True)
            labels_logits.append(gathered_logits[:, 0])
        labels_logits = torch.stack(labels_logits,dim=1)
        return labels_logits


class PromptEncoder(nn.Module):

    def __init__(
        self, 
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        if self.base_model.name_or_path in SUPPORTED_MODELS["decoder_only"]:
            self._forward = self._decoder_only_forward
        elif self.base_model.name_or_path in SUPPORTED_MODELS["encoder_decoder"]:
            self._forward = self._encoder_decoder_forward
        else:
            raise ValueError(f"Architecture type of {self.base_model.name_or_path} model not supported.")

    def _decoder_only_forward(self, encoded_prompts_batch):
        position_ids = self.create_position_ids(encoded_prompts_batch["attention_mask"])
        prompt_output = self.base_model(
            input_ids=encoded_prompts_batch["input_ids"],
            attention_mask=encoded_prompts_batch["attention_mask"],
            position_ids=position_ids,
            use_cache=True, 
            output_attentions=False, 
            output_hidden_states=False
        )
        encoder_output = {
            "input_ids": encoded_prompts_batch["input_ids"],
            "attention_mask": encoded_prompts_batch["attention_mask"],
            "past_key_values": prompt_output.past_key_values,
            "logits": prompt_output.logits
        }
        return encoder_output
    
    def _encoder_decoder_forward(self, encoded_prompts_batch):
        fake_encoded_decoder_input = self.tokenizer([""] * encoded_prompts_batch["input_ids"].shape[0], padding=True, return_tensors="pt")
        fake_encoded_decoder_input = {k: v.to(device=encoded_prompts_batch["input_ids"].device) for k, v in fake_encoded_decoder_input.items()}
        out = self.base_model(
            input_ids=encoded_prompts_batch["input_ids"],
            attention_mask=encoded_prompts_batch["attention_mask"],
            decoder_input_ids=fake_encoded_decoder_input["input_ids"],
            decoder_attention_mask=fake_encoded_decoder_input["attention_mask"],
        )
        encoder_output = {
            "input_ids": encoded_prompts_batch["input_ids"],
            "attention_mask": encoded_prompts_batch["attention_mask"],
            "encoder_last_hidden_state": out.encoder_last_hidden_state,
            "encoder_hidden_states": out.encoder_hidden_states,
            "encoder_attentions": out.encoder_attentions
        }
        return encoder_output


    def forward(self, encoded_prompts_batch):
        encoder_output = self._forward(encoded_prompts_batch)
        return encoder_output

    @staticmethod
    def create_position_ids(attention_mask):
        position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        return position_ids


# class FewShotLanguageModelClassifier(nn.Module):

#     def __init__(
#         self, 
#         base_model: PreTrainedModel,
#         tokenizer: PreTrainedTokenizer, 
#         labels_dict: Dict[int,str],
#     ):
#         super().__init__()
#         self.prompt_encoder = PromptEncoder(base_model, tokenizer)
#         self.labels_decoder = LabelsDecoder(base_model, tokenizer, labels_dict)

#     def forward(self, encoded_prompts_batch):
#         encoder_output = self.prompt_encoder(encoded_prompts_batch)
#         labels_logits = self.labels_decoder(encoder_output)
#         return encoder_output, labels_logits
        
class FewShotLanguageModelClassifier(pl.LightningModule):

    def __init__(
        self, 
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, 
        labels_dict: Dict[int,str],
    ):
        super().__init__()
        self.prompt_encoder = PromptEncoder(base_model, tokenizer)
        self.labels_decoder = LabelsDecoder(base_model, tokenizer, labels_dict)

    def forward(self, encoded_prompts_batch):
        encoder_output = self.prompt_encoder(encoded_prompts_batch)
        labels_logits = self.labels_decoder(encoder_output)
        return encoder_output, labels_logits