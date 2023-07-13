
from ucpa.utils import parse_args, read_config
from tqdm import tqdm





def main():

    # Parse command line arguments and read config file
    args = parse_args()
    config = read_config(args.config_file)



if __name__ == "__main__":
    main()