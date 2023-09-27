from . import *


def load_template(name, **kwargs):
    if name == "simple_query_substitution":
        template = SimpleQuerySubstitutionPrompt(**kwargs)
    elif name == "preface_plus_shots":
        template = PrefacePlusShotsPrompt(**kwargs)
    else:
        raise ValueError(f"Template {name} not supported.")
    return template