import os
import json

def read_config(results_dir):
    with open(os.path.join(results_dir,"config.json"),"r") as f:
        config = json.load(f)
    return config