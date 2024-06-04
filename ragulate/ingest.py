from .utils import load_module, convert_string
from .datasets import get_source_file_paths

from typing import Any, Dict, List

from tqdm import tqdm

from .logging_config import logger

def ingest(
    name: str,
    script_path: str,
    method_name: str,
    var_name: List[str],
    var_value: List[str],
    dataset: List[str],
    **kwargs,
):
    logger.info(
        f"Starting ingest {name} on {script_path}/{method_name} with vars: {var_name} {var_value} on datasets: {dataset}"
    )

    ingest_module = load_module(script_path, "ingest_module")
    ingest_method = getattr(ingest_module, method_name)

    params: Dict[str, Any] = {}
    for i, name in enumerate(var_name):
        params[name] = convert_string(var_value[i])

    source_files = get_source_file_paths("data", datasets=dataset)

    for source_file in tqdm(source_files):
        ingest_method(file_path=source_file, **params)