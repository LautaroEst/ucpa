from .base import SUPPORTED_MODELS
import os
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import glob
import torch

# def get_checkpoint_path(model_name,checkpoints_dir):
#     model_dir = glob.glob(os.path.join(checkpoints_dir,f"{model_name}/models--{model_name.replace('/','--')}/snapshots/**/"))[0]

#     model_checkpoint = None
#     index_file = None
#     index_files = glob.glob(os.path.join(model_dir,"*.index.json"))
#     if len(index_files) == 0:
#         model_checkpoint = glob.glob(os.path.join(model_dir,"*.bin"))
#         if len(model_checkpoint) == 0:
#             model_checkpoint = glob.glob(os.path.join(model_dir,"*.safetensors"))
#             if len(model_checkpoint) == 0:
#                 raise ValueError("No checkpoints files or index file found in the checkpoint directory")
#             elif len(model_checkpoint) == 1:
#                 model_checkpoint = model_checkpoint[0]
#             elif len(model_checkpoint) > 1:
#                 raise ValueError("Multiple checkpoint files found in the checkpoint directory but no index files found")
#         elif len(model_checkpoint) == 1:
#             model_checkpoint = model_checkpoint[0]
#         elif len(model_checkpoint) > 1:
#             raise ValueError("Multiple checkpoint files found in the checkpoint directory but no index files found")
#     elif len(index_files) == 1:
#         index_file = index_files[0]
#     elif len(index_files) > 1:
#         raise ValueError("Multiple index files found in the checkpoint_dir")

#     checkpoint = model_checkpoint if model_checkpoint is not None else index_file
#     return checkpoint


def load_base_model(model_name,checkpoints_dir):

    model_dir = os.path.join(checkpoints_dir,model_name)

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)

    # Load pretrained model
    if model_name in SUPPORTED_MODELS["decoder_only"]:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
        # config = AutoConfig.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config)
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name in SUPPORTED_MODELS["encoder_decoder"]:
        # config = AutoConfig.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
        # with init_empty_weights():
        #     model = AutoModelForSeq2SeqLM.from_config(config)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_dir, local_files_only=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    # checkpoint = get_checkpoint_path(model_name,checkpoints_dir)
    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=checkpoint, device_map="auto", no_split_module_classes=['Block']
    # )

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer