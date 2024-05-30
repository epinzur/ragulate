from llama_index.core.llama_dataset.download import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    download_llama_dataset,
)

import inflection
import os



def handle_download_llama_dataset(
    dataset_name: str,
    download_dir: str,
    llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL,
    llama_datasets_source_files_tree_url: str = LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    **kwargs,
) -> None:

    download_folder = inflection.underscore(dataset_name)
    download_folder = download_folder.removesuffix("_dataset")
    download_dir = os.path.join(download_dir, download_folder)

    if not dataset_name.endswith("Dataset"):
        dataset_name = dataset_name + "Dataset"

    download_llama_dataset(
        llama_dataset_class=dataset_name,
        download_dir=download_dir,
        llama_datasets_lfs_url=llama_datasets_lfs_url,
        llama_datasets_source_files_tree_url=llama_datasets_source_files_tree_url,
        show_progress=True,
        load_documents=False,
    )

    print(f"Successfully downloaded {dataset_name} to {download_dir}")