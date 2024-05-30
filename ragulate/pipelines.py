import importlib.util
import re
from typing import Any, Dict, List

from tqdm import tqdm

from .datasets import get_source_file_paths, get_queries_and_golden_set
from .metrics import metrics
from .logging_config import logger
from trulens_eval.feedback.provider import OpenAI
from trulens_eval import TruChain, Tru

DEFERRED_FEEDBACK_MODE = "deferred"

from .signal_handler import interrupt_received

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

def query(
    name: str,
    script_path: str,
    method_name: str,
    var_name: List[str],
    var_value: List[str],
    dataset: List[str],
    **kwargs,
):
    logger.info(
        f"Starting query {name} on {script_path}/{method_name} with vars: {var_name} {var_value} on datasets: {dataset}"
    )

    Tru(database_url=f"sqlite:///{name}.db") #, name=name)

    query_module = load_module(script_path)
    query_method = getattr(query_module, method_name)

    params: Dict[str, Any] = {}
    for i, name in enumerate(var_name):
        params[name] = convert_string(var_value[i])

    pipeline = query_method(**params)
    llm_provider = OpenAI()

    m = metrics(llm_provider=llm_provider, pipeline=pipeline)

    queries, golden_set = get_queries_and_golden_set("data", datasets=dataset)

    feedback_functions = [
        m.answer_correctness(golden_set=golden_set),
        m.answer_relevance(),
        m.context_relevance(),
        m.groundedness(),
    ]

    recorder = TruChain(
        pipeline,
        app_id=name,
        feedbacks=feedback_functions,
        feedback_mode=DEFERRED_FEEDBACK_MODE,
    )

    for query in tqdm(queries):
        if interrupt_received():
            break
        try:
            with recorder:
                pipeline.invoke(query)
        except:
            print(f"Query: '{query}' caused exception, skipping.")
