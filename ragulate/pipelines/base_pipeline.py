import importlib.util
import re
from abc import ABC
from typing import Any, Dict, List

from ragulate.datasets import BaseDataset


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


class BasePipeline(ABC):
    recipe_name: str
    script_path: str
    method_name: str
    var_names: List[str]
    var_values: List[str]
    datasets: List[BaseDataset]

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        method_name: str,
        var_names: List[str],
        var_values: List[str],
        datasets: List[BaseDataset],
        **kwargs,
    ):
        self.recipe_name = recipe_name
        self.script_path = script_path
        self.method_name = method_name
        self.var_names = var_names
        self.var_values = var_values
        self.datasets = datasets

    def get_method(self, kind: str):
        module = load_module(self.script_path, name=kind)
        return getattr(module, self.method_name)

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for i, name in enumerate(self.var_names):
            params[name] = convert_string(self.var_values[i])
        return params

    def dataset_names(self) -> List[str]:
        return [d.name for d in self.datasets]
