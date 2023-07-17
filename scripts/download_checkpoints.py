
from ucpa.models import SUPPORTED_MODELS
from ucpa.utils import parse_args
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import os


def main():
    args = parse_args()
    checkpoints_dir = args.results_dir
    for architecture_type, models in SUPPORTED_MODELS.items():
        for model in models:
            model_dir = os.path.join(checkpoints_dir,model)
            if os.path.exists(model_dir):
                print(f"Model {model} already downloaded.")
                continue
            print(f"Downloading {model}...")
            if architecture_type == "decoder_only":
                AutoModelForCausalLM.from_pretrained(model, cache_dir=model_dir)
            elif architecture_type == "encoder_decoder":
                AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir=model_dir)
            else:
                raise ValueError(f"Architecture type {architecture_type} not supported.")
            AutoTokenizer.from_pretrained(model, cache_dir=model_dir)
    
if __name__ == '__main__':
    main()


