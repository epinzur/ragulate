import asyncio
import json
from os import path
from typing import Dict, List, Optional, Tuple

from .base_dataset import BaseDataset


class CragDataset(BaseDataset):

    _subset_kinds: List[str] = [
        "aggregation",
        "comparison",
        "false_premise",
        "multi-hop",
        "post-processing",
        "set",
        "simple_w_condition",
        "simple",
    ]

    def __init__(
        self, dataset_name: str, root_storage_path: Optional[str] = "datasets"
    ):
        super().__init__(dataset_name=dataset_name, root_storage_path=root_storage_path)

    def sub_storage_path(self) -> str:
        return path.join("crag", self.name)

    def download_dataset(self) -> None:
        if self.name == "task_1":
            urls = [
                "https://github.com/epinzur/crag_dataset/raw/main/task_1_dev_v4/html_documents.jsonl.bz2",
                "https://github.com/epinzur/crag_dataset/raw/main/task_1_dev_v4/parsed_documents.jsonl.bz2",
                "https://github.com/epinzur/crag_dataset/raw/main/task_1_dev_v4/questions.jsonl.bz2",
            ]
            output_files = [
                path.join(self.storage_path(), "html_documents.jsonl"),
                path.join(self.storage_path(), "parsed_documents.jsonl"),
                path.join(self.storage_path(), "questions.jsonl"),
            ]
            tasks = [self._download_and_decompress(url, output_file) for url, output_file in zip(urls, output_files)]
            asyncio.run(asyncio.gather(*tasks))
        else:
            raise NotImplementedError(f"Crag download not supported for {self.name}")

    def get_source_file_paths(self) -> List[str]:
        raise NotImplementedError("Crag source files are not yet supported")

    def get_queries_and_golden_set(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """gets a list of queries and golden_truth answers for a dataset"""
        queries: List[str] = []
        golden_set: List[Dict[str, str]] = []

        if len(self.subsets) == 0:
            subset_kinds = self._subset_kinds
        else:
            subset_kinds = []
            for subset in self.subsets:
                if subset not in self._subset_kinds:
                    raise ValueError(f"Subset: {subset} doesn't exist in dataset {self.name}. Choices are {self._subset_kinds}")
                subset_kinds.append(subset)

        for subset in subset_kinds:

            json_path = path.join(
                self.storage_path(), f"{subset}.jsonl"
            )
            with open(json_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    query = data.get("query")
                    answer = data.get("answer")
                    if query is not None:
                        queries.append(query)
                        golden_set.append({"query": query, "response": answer})

        return queries, golden_set
