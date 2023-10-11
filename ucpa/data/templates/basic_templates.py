from .utils import ShotsContainer


class SimpleQuerySubstitutionPrompt:
    """ Query substitution template for a zero-shot classification task."""

    def __init__(self, prompt: str, query_pattern="{query}", prefix_sample_separator=" "):
        self.prompt = prompt
        self.pattrn = query_pattern
        self.prefix_sample_separator = prefix_sample_separator

    def construct_prompt(self, query):
        prompt_with_query = self.prompt.replace(self.pattrn, query)
        return prompt_with_query
    

class PrefacePlusShotsPrompt:
    """ Prompt template consisting on a preface followed by n shots (supervised examples). """

    def __init__(self, shots, preface, query_prefix="", label_prefix="", prefix_sample_separator=" ", query_label_separator="\n", shot_separator="\n\n"):
        
        sentences_shots = shots["sentences"]
        labels_shots = shots["labels"]
        
        if sentences_shots is None and labels_shots is None:
            shots_str = ""
        elif sentences_shots is not None and labels_shots is not None:
            if len(sentences_shots) != len(labels_shots):
                raise ValueError("Sentence and label shots must be the same length.")
            shots_str = shot_separator.join(f"{query_prefix}{prefix_sample_separator}{s}{query_label_separator}{label_prefix}{prefix_sample_separator}{l}" for s, l in zip(sentences_shots, labels_shots))
            shots_str = shots_str + shot_separator
        else:
            raise ValueError("Sentence and label shots must either both be None or both be lists of strings.")
        
        self.preface = preface
        self.shots_str = shots_str
        self.query_prefix = query_prefix
        self.label_prefix = label_prefix
        self.prefix_sample_separator = prefix_sample_separator
        self.query_label_separator = query_label_separator
        self.shot_separator = shot_separator
        self.shots = shots

    def construct_prompt(self, query):
        return f"{self.preface}{self.shots_str}{self.query_prefix}{self.prefix_sample_separator}{query}{self.query_label_separator}{self.label_prefix}"
        
    @classmethod
    def from_dataset(cls, data_dir, dataset, split, n_shots, random_state=0, **kwargs):
        if n_shots <= 0:
            raise ValueError("Number of shots must be greater than 0 with this option.")
        shots = ShotsContainer.from_dataset(data_dir, dataset, split, n_shots, random_state)
        return cls(shots, **kwargs)
    
    @classmethod
    def from_lists(cls, sentences, labels, **kwargs):
        shots = ShotsContainer(sentences, labels, **kwargs)
        return cls(shots, **kwargs)


