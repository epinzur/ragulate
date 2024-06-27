from typing import List

from ragulate.datasets import find_dataset
from ragulate.pipelines import QueryPipeline

from ..utils import convert_vars_to_ingredients


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
    query_parser.add_argument(
        "--subset",
        type=str,
        help=(
            "The subset of the dataset to query",
            "Only valid when a single dataset is passed.",
        ),
        action="append",
    )
    query_parser.add_argument(
        "--sample",
        type=float,
        help=(
            "A decimal percentage of the queries to sample for the test",
            "Default is 1.0 (100%)",
        ),
        default=1.0,
    )
    query_parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Random seed to use for query sampling",
            "Ensures reproducibility of tests",
        ),
    )
    query_parser.set_defaults(func=lambda args: call_query(**vars(args)))

    def call_query(
        name: str,
        script_path: str,
        method_name: str,
        var_name: List[str],
        var_value: List[str],
        dataset: List[str],
        subset: List[str],
        sample: float,
        seed: int,
        **kwargs,
    ):
        if sample <= 0.0 or sample > 1.0:
            raise ValueError("Sample percent must be between 0 and 1")

        datasets = [find_dataset(name=name) for name in dataset]

        if subset is not None and len(subset) > 0:
            if len(datasets) > 1:
                raise ValueError(
                    "Only can set `subset` param when there is one dataset"
                )
            else:
                datasets[0].subsets = subset

        ingredients = convert_vars_to_ingredients(
            var_names=var_name, var_values=var_value
        )

        query_pipeline = QueryPipeline(
            recipe_name=name,
            script_path=script_path,
            method_name=method_name,
            ingredients=ingredients,
            datasets=datasets,
            sample_percent=sample,
            random_seed=seed,
        )
        query_pipeline.query()
