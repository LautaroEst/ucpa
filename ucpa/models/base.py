
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import Any, Optional, List, Dict
import lightning.pytorch as pl
import torch.nn.functional as F
from .utils import SUPPORTED_MODELS

class LabelsDecoder(nn.Module):

    def __init__(
        self, 
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, 
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

    def forward(self, encoder_output, encoded_labels):
        labels_logits = []
        for idx in range(len(encoded_labels)):
            encoded_label = encoded_labels[idx]
            logits = self._forward(encoder_output, encoded_label)
            gathered_logprobs = torch.gather(
                torch.log_softmax(logits, dim=-1),
                dim=-1,
                index=encoded_label["input_ids"].unsqueeze(-1)
            ).squeeze(-1).sum(dim=1, keepdim=True)
            labels_logits.append(gathered_logprobs[:, 0])
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

    def _decoder_only_forward(self, batch_encoded_prompts):
        position_ids = self.create_position_ids(batch_encoded_prompts["attention_mask"])
        prompt_output = self.base_model(
            input_ids=batch_encoded_prompts["input_ids"],
            attention_mask=batch_encoded_prompts["attention_mask"],
            position_ids=position_ids,
            use_cache=True, 
            output_attentions=False, 
            output_hidden_states=False
        )
        encoder_output = {
            "input_ids": batch_encoded_prompts["input_ids"],
            "attention_mask": batch_encoded_prompts["attention_mask"],
            "past_key_values": prompt_output.past_key_values,
            "logits": prompt_output.logits
        }
        return encoder_output
    
    def _encoder_decoder_forward(self, batch_encoded_prompts):
        fake_encoded_decoder_input = self.tokenizer([""] * batch_encoded_prompts["input_ids"].shape[0], padding=True, return_tensors="pt")
        out = self.base_model(
            input_ids=batch_encoded_prompts["input_ids"],
            attention_mask=batch_encoded_prompts["attention_mask"],
            decoder_input_ids=fake_encoded_decoder_input["input_ids"].to(device=batch_encoded_prompts["input_ids"].device),
            decoder_attention_mask=fake_encoded_decoder_input["attention_mask"].to(device=batch_encoded_prompts["attention_mask"].device),
        )
        encoder_output = {
            "input_ids": batch_encoded_prompts["input_ids"],
            "attention_mask": batch_encoded_prompts["attention_mask"],
            "encoder_last_hidden_state": out.encoder_last_hidden_state,
            "encoder_hidden_states": out.encoder_hidden_states,
            "encoder_attentions": out.encoder_attentions
        }
        return encoder_output


    def forward(self, batch_encoded_prompts):
        encoder_output = self._forward(batch_encoded_prompts)
        return encoder_output

    @staticmethod
    def create_position_ids(attention_mask):
        position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        return position_ids


class LanguageModelClassifier(pl.LightningModule):

    def __init__(self, model_name):
        super().__init__()

        # Load pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        # Load pretrained model
        if model_name in SUPPORTED_MODELS["decoder_only"]:
            base_model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
            base_model.config.pad_token_id = base_model.config.eos_token_id
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        elif model_name in SUPPORTED_MODELS["encoder_decoder"]:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        else:
            raise ValueError(f"Model {model_name} not supported.")

        self.prompt_encoder = PromptEncoder(base_model, tokenizer)
        self.labels_decoder = LabelsDecoder(base_model, tokenizer)
        self.tokenizer = tokenizer


    def forward(self, batch_encoded_prompts, encoded_labels):
        encoder_output = self.prompt_encoder(batch_encoded_prompts)
        labels_logits = self.labels_decoder(encoder_output, encoded_labels)
        return encoder_output, labels_logits


    def training_step(self, batch, batch_idx, dataloader_idx=0):
        _, labels_logits = self(batch)
        loss = F.cross_entropy(labels_logits,batch["label"])
        return loss


    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()


    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        return super().backward(loss, *args, **kwargs)


    def set_labels_names(self, labels: List[str]):
        labels_dict = {i: label for i, label in enumerate(labels)}
        self.labels_decoder._set_labels_names(labels_dict)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, logits = self(batch["encoded_prompt"], batch["encoded_labels"])
        logits = logits.cpu().numpy()
        labels = batch["label"].cpu().numpy()
        ids = batch["id"]
        prompts = batch["prompt"]
        return ids, prompts, logits, labels
    

        
