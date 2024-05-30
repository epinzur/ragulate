import json
import os
import uuid
from typing import Dict, List, Tuple

from trulens_eval import Feedback, Tru, TruChain, TruLlama
from trulens_eval.app import App
from trulens_eval.feedback.provider import OpenAI

from .framework import Framework
from .metrics import metrics





class Recipe:
    _name: str
    _ingest_pipeline: str
    _query_pipeline: str
    _queries: List[str]

    _tru: Tru

    _temperature = 0
    _feedback_functions = List[Feedback]
    _framework = Framework
    _dataset = "braintrust_coda_help_desk"

    def __init__(self, name: str, framework: Framework) -> None:
        self._name = name
        self._framework = framework

        self._tru = Tru(database_url=f"sqlite://{self._name}.sqlite", name=name)

        llm_provider = OpenAI(model_engine="gpt-3.5-turbo")

        m = metrics(llm_provider=llm_provider, pipeline=self._query_pipeline)

        queries, golden_set = get_queries_and_golden_set_from_llama_index_dataset(
            f"./data/{self._dataset}/"
        )

        self._queries = queries

        self._feedback_functions = [
            m.answer_correctness(golden_set=golden_set),
            m.answer_relevance(),
            m.context_relevance(),
            m.groundedness(),
        ]

    def _get_recorder(self, app_id: str, feedback_mode: str = "deferred"):
        if self._framework == Framework.LANG_CHAIN:
            return TruChain(
                self._query_pipeline,
                app_id=app_id,
                feedbacks=self._feedback_functions,
                feedback_mode=feedback_mode,
            )
        elif self._framework == Framework.LLAMA_INDEX:
            return TruLlama(
                self._query_pipeline,
                app_id=app_id,
                feedbacks=self._feedback_functions,
                feedback_mode=feedback_mode,
            )
        else:
            raise Exception(
                f"Unknown framework: {self._framework} specified for _get_recorder()"
            )

    def _execute_query(self, query):
        if self._framework == Framework.LANG_CHAIN:
            self._query_pipeline.invoke(query)
        elif self._framework == Framework.LLAMA_INDEX:
            self._query_pipeline.query(query)
        else:
            raise Exception(
                f"Unknown framework: {self._framework} specified for execute_query()"
            )

    def cook(self):
        # use a short uuid to ensure that multiple experiments with the same name don't collide in the DB
        shortUuid = str(uuid.uuid4())[9:13]

        app_id = f"{self._name}#{shortUuid}#{self._dataset}"
        tru_recorder = self._get_recorder(app_id=app_id)
        for query in self._queries:
            try:
                with tru_recorder:
                    self._execute_query(query)
            except:
                print(f"Query: '{query}' caused exception, skipping.")
