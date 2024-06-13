from abc import ABC, abstractmethod
from os import path
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BaseDataset(ABC):

    root_storage_path: str
    name: str

    def __init__(
        self, dataset_name: str, root_storage_path: Optional[str] = "datasets"
    ):
        self.name = dataset_name
        self.root_storage_path = root_storage_path

    def storage_path(self) -> str:
        """returns the path where dataset files should be stored"""
        return path.join(self.root_storage_path, self.sub_storage_path())

    def list_files_at_path(self, path: str) -> List[str]:
        """lists all files at a path (excluding dot files)"""
        return [
            f
            for f in Path(path).iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

    @abstractmethod
    def sub_storage_path(self) -> str:
        """the sub-path to store the dataset in"""

    @abstractmethod
    def download_dataset(self):
        """downloads a dataset locally"""

    @abstractmethod
    def get_source_file_paths(self) -> List[str]:
        """gets a list of source file paths for for a dataset"""

    @abstractmethod
    def get_queries_and_golden_set(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """gets a list of queries and golden_truth answers for a dataset"""
