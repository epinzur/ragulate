from typing import Any, Dict

from tqdm import tqdm

from ..logging_config import logger
from .base_pipeline import BasePipeline, convert_string, load_module


class IngestPipeline(BasePipeline):
    def ingest(self):
        logger.info(
            f"Starting ingest {self.recipe_name} on {self.script_path}/{self.method_name} with vars: {self.var_names} {self.var_values} on datasets: {self.dataset_names()}"
        )

        ingest_module = load_module(self.script_path, "ingest_module")
        ingest_method = getattr(ingest_module, self.method_name)

        params: Dict[str, Any] = {}
        for i, name in enumerate(self.var_names):
            params[name] = convert_string(self.var_values[i])

        source_files = []
        for dataset in self.datasets:
            source_files.extend(dataset.get_source_file_paths())

        for source_file in tqdm(source_files):
            ingest_method(file_path=source_file, **params)
