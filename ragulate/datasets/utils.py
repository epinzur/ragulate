from typing import List

from .base_dataset import BaseDataset
from .llama_dataset import LlamaDataset

# TODO: implement this when adding additional dataset kinds
def find_dataset(name:str) -> BaseDataset:
    """ searches for a downloaded dataset with this name. if found, returns it."""
    return get_dataset(name, "llama")

def get_dataset(name:str, kind:str) -> BaseDataset:
    if kind == "llama":
        return LlamaDataset(dataset_name=name)

    raise NotImplementedError("only llama datasets are currently supported")

