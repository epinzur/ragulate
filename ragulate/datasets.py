import os
from pathlib import Path
from typing import List

import inflection
from llama_index.core.llama_dataset import download
from llama_index.core.llama_dataset.download import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
)


def get_llama_dataset_path(dataset_name: str, base_path: str) -> str:
    folder = inflection.underscore(dataset_name)
    folder = folder.removesuffix("_dataset")
    return os.path.join(base_path, folder)


def download_llama_dataset(
    dataset_name: str,
    download_dir: str,
    llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL,
    llama_datasets_source_files_tree_url: str = LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    **kwargs,
) -> None:

    download_path = get_llama_dataset_path(
        dataset_name=dataset_name, base_path=download_dir
    )

    if not dataset_name.endswith("Dataset"):
        dataset_name = dataset_name + "Dataset"

    download.download_llama_dataset(
        llama_dataset_class=dataset_name,
        download_dir=download_path,
        llama_datasets_lfs_url=llama_datasets_lfs_url,
        llama_datasets_source_files_tree_url=llama_datasets_source_files_tree_url,
        show_progress=True,
        load_documents=False,
    )

    print(f"Successfully downloaded {dataset_name} to {download_dir}")


def get_source_file_paths(base_path: str, datasets: List[str]) -> List[str]:
    file_paths = []

    for dataset in datasets:
        source_path = os.path.join(
            get_llama_dataset_path(dataset_name=dataset, base_path=base_path),
            "source_files",
        )

        file_paths.extend([f for f in Path(source_path).iterdir() if f.is_file()])

    return file_paths
