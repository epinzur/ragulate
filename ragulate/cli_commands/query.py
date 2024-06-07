from typing import List

from ragulate.datasets import load_datasets
from ragulate.pipelines import QueryPipeline


def setup_query(subparsers):
    query_parser = subparsers.add_parser("query", help="Run an query pipeline")
    query_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="A unique name for the query pipeline",
        required=True,
    )
    query_parser.add_argument(
        "-s",
        "--script_path",
        type=str,
        help="The path to the python script that contains the query method",
        required=True,
    )
    query_parser.add_argument(
        "-m",
        "--method-name",
        type=str,
        help="The name of the method in the script to run query",
        required=True,
    )
    query_parser.add_argument(
        "--var-name",
        type=str,
        help=(
            "The name of a variable in the query script",
            "This should be paired with a `--var-value` argument",
            "and can be passed multiple times.",
        ),
        action="append",
    )
    query_parser.add_argument(
        "--var-value",
        type=str,
        help=(
            "The value of a variable in the query script",
            "This should be paired with a `--var-name` argument",
            "and can be passed multiple times.",
        ),
        action="append",
    )
    query_parser.add_argument(
        "--dataset",
        type=str,
        help=("The name of a dataset to query", "This can be passed multiple times."),
        action="append",
    )
    query_parser.set_defaults(func=lambda args: call_query(**vars(args)))

    def call_query(
        name: str,
        script_path: str,
        method_name: str,
        var_name: List[str],
        var_value: List[str],
        dataset: List[str],
        **kwargs,
    ):
        datasets = load_datasets(dataset_names=dataset)

        query_pipeline = QueryPipeline(
            recipe_name=name,
            script_path=script_path,
            method_name=method_name,
            var_names=var_name,
            var_values=var_value,
            datasets=datasets,
        )
        query_pipeline.query()
