import importlib.util
import re
from typing import Any, Dict, List

from tqdm import tqdm

from .datasets import get_source_file_paths
from .logging_config import logger


def convert_string(s):
    s = s.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    elif re.match(r"^\d*\.\d+$", s):
        return float(s)
    else:
        return s


# Function to dynamically load a module
def load_module(file_path):
    spec = importlib.util.spec_from_file_location("ingest_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    ingest_module = load_module(script_path)
    ingest_method = getattr(ingest_module, method_name)

    params: Dict[str, Any] = {}
    for i, name in enumerate(var_name):
        params[name] = convert_string(var_value[i])

    source_files = get_source_file_paths("data", datasets=dataset)

    for source_file in tqdm(source_files):
        ingest_method(file_path=source_file, **params)
