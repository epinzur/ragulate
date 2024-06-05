
from typing import Any, Dict, List

from tqdm import tqdm

from .datasets import get_queries_and_golden_set
from .metrics import metrics
from .logging_config import logger
from trulens_eval.feedback.provider import OpenAI
from trulens_eval import TruChain, Tru
from trulens_eval.schema.feedback import FeedbackResultStatus
import signal
import time

from .utils import load_module, convert_string, get_tru

DEFERRED_FEEDBACK_MODE = "deferred"


class QueryPipeline:
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

    def __init__(self, name: str, datasets: List[str]):
        self._name = name
        self._tru = get_tru(recipe_name=name)
        self._tru.reset_database()

        # Set up the signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.signal_handler)

        for dataset in datasets:
            self._queries[dataset], self._golden_sets[dataset] = get_queries_and_golden_set("data", dataset=dataset)
            self._total_queries += len(self._queries[dataset])

        metric_count = 4
        self._total_feedbacks = self._total_queries * metric_count

    def signal_handler(self, sig, frame):
        self._sigint_received = True
        self.stop_evaluation("sigint")


    def start_evaluation(self):
        self._tru.start_evaluator(disable_tqdm=True)
        self._evaluation_running = True

    def stop_evaluation(self, loc:str):
        if self._evaluation_running:
            try:
                print(f"Stopping evaluation from: {loc}")
                self._tru.stop_evaluator()
                self._evaluation_running = False
            except Exception as e:
                print(f"issue stopping evaluator: {e}")
            finally:
                self._progress.close()

    def update_progress(self, query_change:int = 0):
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

    def query(
        self,
        script_path: str,
        method_name: str,
        var_names: List[str],
        var_values: List[str],
        datasets: List[str],
        **kwargs,
    ):

        query_module = load_module(script_path, name="query_module")
        query_method = getattr(query_module, method_name)

        params: Dict[str, Any] = {}
        for i, name in enumerate(var_names):
            params[name] = convert_string(var_values[i])

        pipeline = query_method(**params)
        llm_provider = OpenAI()

        m = metrics(llm_provider=llm_provider, pipeline=pipeline)

        self.start_evaluation()

        time.sleep(0.1)
        print(
            f"Starting query {self._name} on {script_path}/{method_name} with vars: {var_names} {var_values} on datasets: {datasets}"
        )
        print("Progress postfix legend: (q)ueries completed; Evaluations (d)one, (r)unning, (w)aiting, (f)ailed, (s)kipped")

        self._progress = tqdm(total=(self._total_queries + self._total_feedbacks))

        for dataset in self._queries:
            feedback_functions = [
                m.answer_correctness(golden_set=self._golden_sets[dataset]),
                m.answer_relevance(),
                m.context_relevance(),
                m.groundedness(),
            ]

            recorder = TruChain(
                pipeline,
                app_id=dataset,
                feedbacks=feedback_functions,
                feedback_mode=DEFERRED_FEEDBACK_MODE,
            )

            for query in self._queries[dataset]:
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
