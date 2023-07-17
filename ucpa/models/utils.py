from .base import SUPPORTED_MODELS
import os
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

def load_base_model(model_name,checkpoints_dir):
    model_dir = os.path.join(checkpoints_dir,model_name)
    if model_name in SUPPORTED_MODELS["decoder_only"]:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
    elif model_name in SUPPORTED_MODELS["encoder_decoder"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
    return model, tokenizer