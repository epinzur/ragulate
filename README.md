# ragulate

A tool for evaluating RAG pipelines

![](logo_smaller.png)


## Installation

```sh
pip install ragulate
```

## Usage

```sh
usage: ragulate [-h] {download-llamadataset,ingest} ...

RAGu-late CLI tool.

options:
  -h, --help            show this help message and exit

commands:
  {download-llamadataset,ingest}
    download-llamadataset
                        Download a llama-dataset
    ingest              Run an ingest pipeline
```

### Download Dataset Example

```
ragulate download-llamadataset BraintrustCodaHelpDesk
```

### Ingest Example

This command should work:
```
ragulate ingest -n chunk_size_500 -s experiment_chunk_size_and_k.py -m ingest --var-name chunk_size --var-value 500 --dataset BraintrustCodaHelpDesk
```
