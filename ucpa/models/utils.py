from .base import SUPPORTED_MODELS
import os
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def load_base_model(model_name):

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # Load pretrained model
    if model_name in SUPPORTED_MODELS["decoder_only"]:
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name in SUPPORTED_MODELS["encoder_decoder"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model, tokenizer