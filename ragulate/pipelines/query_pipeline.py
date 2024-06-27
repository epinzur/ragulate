import signal
import time
<<<<<<< HEAD
from typing import Dict, List, Optional
=======
from typing import Any, Dict, List
>>>>>>> main

from tqdm import tqdm
from trulens_eval import Tru, TruChain
from trulens_eval.feedback.provider import OpenAI, AzureOpenAI, Bedrock, LiteLLM, Langchain, Huggingface
from trulens_eval.schema.feedback import FeedbackMode, FeedbackResultStatus

from ragulate.datasets import BaseDataset

from ..logging_config import logger
from ..utils import get_tru
from .base_pipeline import BasePipeline
from .feedbacks import Feedbacks

import random
import numpy as np

class QueryPipeline(BasePipeline):
    _sigint_received = False

    _tru: Tru
    _name: str
    _progress: tqdm
    _queries: Dict[str, List[str]] = {}
    _golden_sets: Dict[str, List[Dict[str, str]]] = {}
    _total_queries: int = 0
    _total_feedbacks: int = 0
    _finished_feedbacks: int = 0
    _finished_queries: int = 0
    _evaluation_running = False

    @property
    def PIPELINE_TYPE(self):
        return "query"

    @property
    def get_reserved_params(self) -> List[str]:
        return []

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        method_name: str,
        ingredients: Dict[str, Any],
        datasets: List[BaseDataset],
        sample: str = None,
        seed: str = '0',
        llm_provider: str = OpenAI,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            recipe_name=recipe_name,
            script_path=script_path,
            method_name=method_name,
            ingredients=ingredients,
            datasets=datasets,
        )
        # Set up the signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.signal_handler)

        for dataset in datasets:
            self._queries[dataset.name], self._golden_sets[dataset.name] = (
                dataset.get_queries_and_golden_set()
            )
            self._total_queries += len(self._queries[dataset.name])

        metric_count = 4
        self._total_feedbacks = self._total_queries * metric_count

    def signal_handler(self, sig, frame):
        self._sigint_received = True
        self.stop_evaluation("sigint")

    def start_evaluation(self):
        self._tru = get_tru(recipe_name=self.recipe_name)
        self._tru.reset_database()
        self._tru.start_evaluator(disable_tqdm=True)
        self._evaluation_running = True

    def stop_evaluation(self, loc: str):
        if self._evaluation_running:
            try:
                logger.debug(f"Stopping evaluation from: {loc}")
                self._tru.stop_evaluator()
                self._evaluation_running = False
                self._tru.delete_singleton()
            except Exception as e:
                logger.error(f"issue stopping evaluator: {e}")
            finally:
                self._progress.close()

    def update_progress(self, query_change: int = 0):
        self._finished_queries += query_change

        status = self._tru.db.get_feedback_count_by_status()
        done = status.get(FeedbackResultStatus.DONE, 0)

        postfix = {
            "q": self._finished_queries,
            "d": done,
            "r": status.get(FeedbackResultStatus.RUNNING, 0),
            "w": status.get(FeedbackResultStatus.NONE, 0),
            "f": status.get(FeedbackResultStatus.FAILED, 0),
            "s": status.get(FeedbackResultStatus.SKIPPED, 0),
        }
        self._progress.set_postfix(postfix)

        update = query_change + (done - self._finished_feedbacks)
        if update > 0:
            self._progress.update(update)

        self._finished_feedbacks = done
    
    def initialize_provider(self, provider_name: str, model_name: str):
        if model_name == None:
            if provider_name == 'OpenAI':
                return OpenAI()
            elif provider_name == 'AzureOpenAI':
                return AzureOpenAI()
            elif provider_name == 'Bedrock':
                return Bedrock()
            elif provider_name == 'LiteLLM':
                return LiteLLM()
            elif provider_name == 'Langchain':
                return Langchain()
            elif provider_name == 'Huggingface':
                return Huggingface()
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")
        else:
            if provider_name == 'OpenAI':
                return OpenAI(model_engine=model_name)
            elif provider_name == 'AzureOpenAI':
                return AzureOpenAI(deployment_name=model_name)
            elif provider_name == 'Bedrock':
                return Bedrock(model_id=model_name)
            elif provider_name == 'LiteLLM':
                return LiteLLM(model_engine=model_name)
            elif provider_name == 'Langchain':
                return Langchain(chain=model_name)
            elif provider_name == 'Huggingface':
                return Huggingface(name=model_name)
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")
    

    def load_and_sample_queries(self, dataset):
        queries = dataset.get_queries()
        golden_set = dataset.get_golden_set()

        if self._sample:
            sample_indices = np.random.choice(len(queries), self._sample, replace=False)
            sampled_queries = [queries[i] for i in sample_indices]
            sampled_golden_set = [golden_set[i] for i in sample_indices]
            return sampled_queries, sampled_golden_set
        return queries, golden_set
    
    def query(self):
        query_method = self.get_method()

        pipeline = query_method(**self.ingredients)
        llm_provider = OpenAI(model_engine="gpt-3.5-turbo")

        feedbacks = Feedbacks(llm_provider=llm_provider, pipeline=pipeline)

        self.start_evaluation()

        time.sleep(0.1)
        logger.info(
            f"Starting query {self.recipe_name} on {self.script_path}/{self.method_name} with ingredients: {self.ingredients} on datasets: {self.dataset_names()}"
        )
        logger.info(
            "Progress postfix legend: (q)ueries completed; Evaluations (d)one, (r)unning, (w)aiting, (f)ailed, (s)kipped"
        )

        self._progress = tqdm(total=(self._total_queries + self._total_feedbacks))
        
        
        
        for dataset_name in self._queries:
            feedback_functions = [
                feedbacks.answer_correctness(
                    golden_set=self._golden_sets[dataset_name]
                ),
                feedbacks.answer_relevance(),
                feedbacks.context_relevance(),
                feedbacks.groundedness(),
            ]

            recorder = TruChain(
                pipeline,
                app_id=dataset_name,
                feedbacks=feedback_functions,
                feedback_mode=FeedbackMode.DEFERRED,
            )
            
            if self._sample:
                
                if self._sample < 1:
                    self._sample = self._sample*len(self._queries[dataset_name])
                sample_indices = np.random.choice(len(self._queries), self._sample, replace=False)
                sampled_queries = [self._queries[i] for i in sample_indices]

            for query in self._queries[dataset_name]:
                if self._sigint_received:
                    break
                try:
                    with recorder:
                        pipeline.invoke(query)
                except Exception as e:
                    # TODO: figure out why the logger isn't working after tru-lens starts. For now use print()
                    print(
                        f"ERROR: Query: '{query}' caused exception, skipping. Exception {e}"
                    )
                    logger.error(f"Query: '{query}' caused exception: {e}, skipping.")
                finally:
                    self.update_progress(query_change=1)

        while self._finished_feedbacks < self._total_feedbacks:
            if self._sigint_received:
                break
            self.update_progress()
            time.sleep(1)

        self.stop_evaluation(loc="end")
