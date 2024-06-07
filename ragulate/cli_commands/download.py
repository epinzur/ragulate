from ragulate.datasets import LlamaDataset


def setup_download(subparsers):
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument(
        "dataset_name",
        type=str,
        help=(
            "The name of the dataset you want to download, "
            "such as `PaulGrahamEssayDataset`."
        ),
    )
    download_parser.add_argument(
        "-k",
        "--kind",
        type=str,
        help="The kind of dataset to download. Currently only `llama` is supported",
        required=True,
    )
    download_parser.set_defaults(func=lambda args: call_download(**vars(args)))


def call_download(dataset_name: str, kind: str, **kwargs):
    if not kind == "llama":
        raise ("Currently only Llama Datasets are supported. Set param `-k llama`")
    llama = LlamaDataset(dataset_name=dataset_name)
    llama.download_dataset()
