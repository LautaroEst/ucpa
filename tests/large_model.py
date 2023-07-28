

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_in_model, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from utils import parse_args
from ucpa.models import load_base_model
import os
import glob

# model_name = "gpt2"
# model_name = "gpt2-xl"
model_name = "meta-llama/Llama-2-7b-hf"
modules = [
    "transformer.wte",
    "transformer.wpe",
    *[f"transformer.h.{i}" for i in range(12)],
    "lm_head"
]
# device_map = {
#     "transformer.wte": "cuda:0",
#     "transformer.wpe": "cuda:0",
#     **{f"transformer.h.{i}": "disk" for i in range(4)},
#     **{f"transformer.h.{i}":"cpu" for i in range(4,8)},
#     **{f"transformer.h.{i}":"cuda:0" for i in range(8,12)},
#     "lm_head": "cuda:0"
# }
device_map = {
    "transformer.wte": "cuda:0",
    "transformer.wpe": "cuda:0",
    **{f"h.{i}": "disk" for i in range(4)},
    **{f"h.{i}":"cpu" for i in range(4,8)},
    **{f"h.{i}":"cuda:0" for i in range(8,12)},
    "lm_head": "cuda:0"
}
   

def main():

    args = parse_args()
    model_dir = glob.glob(os.path.join(args.checkpoints_dir,f"{model_name}/models--{model_name.replace('/','--')}/snapshots/**/"))[0]


    import safetensors
    with safetensors.safe_open(os.path.join(model_dir,"model-00001-of-00002.safetensors"), framework="pt", device=0) as f:
        keys = f.keys()
    print(keys)


    # model, tokenizer = load_base_model(model_name,args.checkpoints_dir)

    # print(model)


    
    # device_map = infer_auto_device_map(model, max_memory={"cuda:0": "500MB", "cpu": "500MB"})
    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=checkpoint, device_map="auto", no_split_module_classes=['Block']
    # )
    # print(model)

    # load_checkpoint_in_model(
    #     model,
    #     checkpoint,
    #     device_map=device_map,
    #     offload_folder="results/tmp",
    #     offload_state_dict=True,
    #     offload_buffers=True
    # )
    # model = dispatch_model(
    #     model,
    #     device_map=device_map,
    #     main_device="cuda:0",
    #     state_dict=None,
    #     offload_dir="results/tmp",
    #     offload_index=None,
    #     offload_buffers=True,
    #     skip_keys=None,
    #     preload_module_classes=None
    # )

    # encoded_input = {
    #     "input_ids": torch.randint(0, 100, (2, 1024)),
    #     "attention_mask": torch.ones((2, 1024)),
    # }
    # y = model(
    #     input_ids=encoded_input["input_ids"],
    #     attention_mask=encoded_input["attention_mask"],
    # )
    




if __name__ == '__main__':
    main()
    
