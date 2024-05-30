import argparse
from typing import Any, Optional

from .datasets import handle_download_llama_dataset, LLAMA_DATASETS_LFS_URL

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGu-late CLI tool.")

    # Subparsers for the main commands
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # download llamadatasets command
    llamadataset_parser = subparsers.add_parser(
        "download-llamadataset", help="Download a llama-dataset"
    )
    llamadataset_parser.add_argument(
        "dataset_name",
        type=str,
        help=(
            "The name of the llama-dataset you want to download, "
            "such as `PaulGrahamEssayDataset`."
        ),
    )
    llamadataset_parser.add_argument(
        "-d",
        "--download-dir",
        type=str,
        default="./data",
        help="Custom dirpath to download the dataset into.",
    )
    llamadataset_parser.add_argument(
        "--llama-datasets-lfs-url",
        type=str,
        default=LLAMA_DATASETS_LFS_URL,
        help="URL to llama datasets.",
    )
    llamadataset_parser.set_defaults(
        func=lambda args: handle_download_llama_dataset(**vars(args))
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    args.func(args)


if __name__ == "__main__":
    main()