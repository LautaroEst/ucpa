import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data"
    )
    return parser.parse_args()