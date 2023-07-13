

from ucpa.utils import parse_args, read_config
from ucpa.data import PromptTemplate, ClassificationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

QUERY_LABEL_SEPARATOR = "\n"
SHOT_SEPARATOR = "\n\n"


dataset_config = {

    "trec": (PromptTemplate(
        preface="Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n",
        query_prefix="Question: ",
        label_prefix="Answer Type: ",
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'Number', 1: 'Location', 2: 'Person', 3: 'Description', 4: 'Entity', 5: 'Abbreviation'}),
    
    "sst2": (PromptTemplate(
        preface="",
        query_prefix="Review: ",
        label_prefix="Sentiment: ",
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'Negative', 1: 'Positive'}),
    
    "agnews": (PromptTemplate(
        preface="Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n",
        query_prefix="Article: ",
        label_prefix="Answer: ",
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}),
    
    "dbpedia": (PromptTemplate(
        preface="Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n",
        query_prefix="Article: ",
        label_prefix="Answer: ",
        query_label_separator=QUERY_LABEL_SEPARATOR,
        shot_separator=SHOT_SEPARATOR
    ), {0: 'Company', 1: 'School', 2: 'Artist', 3: 'Athlete', 4: 'Politician', 5: 'Transportation', 6: 'Building', 7: 'Nature', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'Book'}),

}


def create_dataloader(dataset_name,split,data_dir,template,label_dict,num_subsamples=None,random_state=None):
    dataset = ClassificationDataset(dataset_name,split,data_dir,template,label_dict)
    dataset.set_shots(n_shots=0)
    if num_subsamples is not None:
        dataset.set_subsamples(num_subsamples,random_state)



def main():

    # Parse command line arguments and read config file
    args = parse_args()
    config = read_config(args.config_file)

    # Iterate over datasets
    dataset_bar = tqdm(config["datasets"], desc="Dataset")
    for dataset_name in dataset_bar:
        dataset_bar.set_description(f"Dataset: {dataset_name}")
        template, label_dict = dataset_config[dataset_name]
        train_dataset = ClassificationDataset(dataset_name,"train",args.data_dir,template,label_dict)
        train_dataset.set_shots(n_shots=0)
        test_dataset = ClassificationDataset(dataset_name,"test",args.data_dir,template,label_dict)
        test_dataset.set_shots(n_shots=0)
        






if __name__ == "__main__":
    main()