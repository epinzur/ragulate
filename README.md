# ragulate

A tool for evaluating RAG pipelines

![](logo_smaller.png)


## Installation

```sh
pip install ragulate
```

## Usage

```sh
usage: ragulate [-h] {download-llamadataset,ingest,query} ...

RAGu-late CLI tool.

options:
  -h, --help            show this help message and exit

commands:
  {download-llamadataset,ingest,query}
    download-llamadataset
                        Download a llama-dataset
    ingest              Run an ingest pipeline
    query               Run an query pipeline
```

### Download Dataset Example

```
ragulate download-llamadataset BraintrustCodaHelpDesk
```

### Ingest Example

This command should work:
```
ragulate ingest -n chunk_size_500_k_2 -s experiment_chunk_size_and_k.py -m ingest \
--var-name chunk_size --var-value 500 --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk
```

### Query Exmaple

This command should work:
```
ragulate query -n chunk_size_500_k_2 -s experiment_chunk_size_and_k.py -m query_pipeline \
--var-name chunk_size --var-value 500  --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk
```
