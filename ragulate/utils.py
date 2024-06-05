import importlib.util
import re

from trulens_eval import TruChain, Tru

def convert_string(s):
    s = s.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    elif re.match(r"^\d*\.\d+$", s):
        return float(s)
    else:
        return s


# Function to dynamically load a module
def load_module(file_path, name):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_tru(recipe_name: str) -> Tru:
    return Tru(database_url=f"sqlite:///{recipe_name}.sqlite") #, name=name)