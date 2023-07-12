
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="./results"
    )
    parser.add_argument(
        "--config-file", 
        type=str
    )
    return parser.parse_args()