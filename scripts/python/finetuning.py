
from ucpa.utils import parse_args


def main():

    # Parse command line arguments
    args = parse_args()
    print(args.data_dir)
    print(args.results_dir)
    print(args.config_file)


if __name__ == "__main__":
    main()