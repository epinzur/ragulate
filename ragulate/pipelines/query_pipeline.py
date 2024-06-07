import signal
import time
from typing import Dict, List

from tqdm import tqdm
from trulens_eval import Tru, TruChain
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.schema.feedback import FeedbackMode, FeedbackResultStatus

from ragulate.datasets import BaseDataset

from ..logging_config import logger
from ..utils import get_tru
from .base_pipeline import BasePipeline
from .feedbacks import Feedbacks


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
        super().__init__(
            recipe_name=recipe_name,
            script_path=script_path,
            method_name=method_name,
            var_names=var_names,
            var_values=var_values,
            datasets=datasets,
        )
        self._tru = get_tru(recipe_name=recipe_name)
        self._tru.reset_database()

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
        self._tru.start_evaluator(disable_tqdm=True)
        self._evaluation_running = True

    def stop_evaluation(self, loc: str):
        if self._evaluation_running:
            try:
                logger.debug(f"Stopping evaluation from: {loc}")
                self._tru.stop_evaluator()
                self._evaluation_running = False
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

    def query(self):
        query_method = self.get_method(kind="query")
        params = self.get_params()

        pipeline = query_method(**params)
        llm_provider = OpenAI()

        feedbacks = Feedbacks(llm_provider=llm_provider, pipeline=pipeline)

        self.start_evaluation()

        time.sleep(0.1)
        logger.info(
            f"Starting query {self.recipe_name} on {self.script_path}/{self.method_name} with vars: {self.var_names} {self.var_values} on datasets: {self.dataset_names()}"
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

            for query in self._queries[dataset_name]:
                if self._sigint_received:
                    break
                try:
                    with recorder:
                        pipeline.invoke(query)
                except Exception as e:
                    logger.error(f"Query: '{query}' caused exception, skipping.")
                finally:
                    self.update_progress(query_change=1)

        while self._finished_feedbacks < self._total_feedbacks:
            if self._sigint_received:
                break
            self.update_progress()
            time.sleep(1)

        self.stop_evaluation(loc="end")
