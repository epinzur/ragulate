import argparse

from dotenv import load_dotenv

from .datasets import LLAMA_DATASETS_LFS_URL, download_llama_dataset
from .ingest import ingest
from .logging_config import logger

if load_dotenv():
    logger.info("Parsed .env file successfully")


def setup_download_llama_dataset(subparsers):
    download_parser = subparsers.add_parser(
        "download-llamadataset", help="Download a llama-dataset"
    )
    download_parser.add_argument(
        "dataset_name",
        type=str,
        help=(
            "The name of the llama-dataset you want to download, "
            "such as `PaulGrahamEssayDataset`."
        ),
    )
    download_parser.add_argument(
        "-d",
        "--download-dir",
        type=str,
        default="./data",
        help="Custom dirpath to download the dataset into.",
    )
    download_parser.add_argument(
        "--llama-datasets-lfs-url",
        type=str,
        default=LLAMA_DATASETS_LFS_URL,
        help="URL to llama datasets.",
    )
    download_parser.set_defaults(func=lambda args: download_llama_dataset(**vars(args)))


def setup_ingest(subparsers):
    ingest_parser = subparsers.add_parser("ingest", help="Run an ingest pipeline")
    ingest_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="A unique name for the ingest pipeline",
        required=True,
    )
    ingest_parser.add_argument(
        "-s",
        "--script_path",
        type=str,
        help="The path to the python script that contains the ingest method",
        required=True,
    )
    ingest_parser.add_argument(
        "-m",
        "--method-name",
        type=str,
        help="The name of the method in the script to run ingest",
        required=True,
    )
    ingest_parser.add_argument(
        "--var-name",
        type=str,
        help=(
            "The name of a variable in the ingest script",
            "This should be paired with a `--var-value` argument",
            "and can be passed multiple times.",
        ),
        action="append",
    )
    ingest_parser.add_argument(
        "--var-value",
        type=str,
        help=(
            "The value of a variable in the ingest script",
            "This should be paired with a `--var-name` argument",
            "and can be passed multiple times.",
        ),
        action="append",
    )
    ingest_parser.add_argument(
        "--dataset",
        type=str,
        help=("The name of a dataset to ingest", "This can be passed multiple times."),
        action="append",
    )
    ingest_parser.set_defaults(func=lambda args: ingest(**vars(args)))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGu-late CLI tool.")

    # Subparsers for the main commands
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    setup_download_llama_dataset(subparsers=subparsers)
    setup_ingest(subparsers=subparsers)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    args.func(args)


if __name__ == "__main__":
    main()
