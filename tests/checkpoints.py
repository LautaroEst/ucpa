from utils import parse_args
from ucpa.models import load_base_model

def main():
    args = parse_args()
    model, tokenizer = load_base_model("gpt2",args.checkpoints_dir)
    print(model)
    print(tokenizer)



if __name__ == '__main__':
    main()