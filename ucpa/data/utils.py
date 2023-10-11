from .datasets import *
from .templates import *

def load_dataset(name, *args, **kwargs):
    dataset_cls = dataset_name2class[name]
    return dataset_cls(*args, **kwargs)


def load_template(name, **kwargs):
    if name == "simple_query_substitution":
        template = SimpleQuerySubstitutionPrompt(**kwargs)
    elif name == "preface_plus_shots":
        if "sample_shots_from" in kwargs:
            all_kwargs = {
                **kwargs["sample_shots_from"],
                **{k: v for k, v in kwargs.items() if k != "sample_shots_from"}
            }
            template = PrefacePlusShotsPrompt.from_dataset(**all_kwargs)
        elif "shots" in kwargs:
            template = PrefacePlusShotsPrompt.from_lists(**kwargs["shots"])
        else:
            raise ValueError("preface_plus_shots option should contain 'sample_shots_from' or 'shots' option.")
    else:
        raise ValueError(f"Template {name} not supported.")
    return template


