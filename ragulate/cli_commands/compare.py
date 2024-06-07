from typing import List

from ..analysis import Analysis


def setup_compare(subparsers):
    compare_parser = subparsers.add_parser(
        "compare", help="Compare results from 2 (or more) recipes"
    )
    compare_parser.add_argument(
        "-r",
        "--recipe",
        type=str,
        help="A recipe to compare. This can be passed multiple times.",
        required=True,
        action="append",
    )
    compare_parser.set_defaults(func=lambda args: call_compare(**vars(args)))


def call_compare(
    recipe: List[str],
    **kwargs,
):
    analysis = Analysis()
    analysis.compare(recipes=recipe)
