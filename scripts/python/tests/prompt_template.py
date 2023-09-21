
from utils import parse_args
from ucpa.data import PromptTemplate


dataset2template = {
    "trec": PromptTemplate(
        preface="Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n",
        query_prefix="Question: ",
        label_prefix="Answer Type: ",
        query_label_separator="\n",
        shot_separator="\n\n"
    ),
}

def main():
    # args = parse_args()
    # print(args.data_dir)
    
    template = dataset2template["trec"]
    prompt = template.construct_prompt("What is the capital of France?",["Paris","Paris","Paris"],["Location","Location","Location"])
    print([prompt])


if __name__ == '__main__':
    main()