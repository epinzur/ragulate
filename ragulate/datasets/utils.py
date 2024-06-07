from typing import List

from .base_dataset import BaseDataset
from .llama_dataset import LlamaDataset


def load_datasets(dataset_names: List[str]) -> List[BaseDataset]:
    return [LlamaDataset(dataset_name=name) for name in dataset_names]
