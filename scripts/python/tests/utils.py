import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=""
    )
    parser.add_argument(
        "--checkpoints-dir", 
        type=str, 
        default=""
    )
    return parser.parse_args()