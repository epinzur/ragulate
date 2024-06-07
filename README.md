# ragulate

A tool for evaluating RAG pipelines

![](images/logo_smaller.png)


## Installation

```sh
pip install ragulate
```

## Usage

```sh
usage: ragulate [-h] {download,ingest,query,compare} ...

RAGu-late CLI tool.

options:
  -h, --help            show this help message and exit

commands:
    download            Download a dataset
    ingest              Run an ingest pipeline
    query               Run an query pipeline
    compare             Compare results from 2 (or more) recipes
```

### Download Dataset Example

```
ragulate download -k llama BraintrustCodaHelpDesk
```

### Ingest Example

These commands should work:
```
ragulate ingest -n chunk_size_500_k_2 -s experiment_chunk_size_and_k.py -m ingest \
--var-name chunk_size --var-value 500 --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk

ragulate ingest -n chunk_size_1000_k_2 -s experiment_chunk_size_and_k.py -m ingest \
--var-name chunk_size --var-value 1000 --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk
```

### Query Exmaple

These commands should work:
```
ragulate query -n chunk_size_500_k_2 -s experiment_chunk_size_and_k.py -m query_pipeline \
--var-name chunk_size --var-value 500  --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk

ragulate query -n chunk_size_1000_k_2 -s experiment_chunk_size_and_k.py -m query_pipeline \
--var-name chunk_size --var-value 1000  --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk

ragulate query -n chunk_size_500_k_5 -s experiment_chunk_size_and_k.py -m query_pipeline \
--var-name chunk_size --var-value 500  --var-name k --var-value 5 --dataset BraintrustCodaHelpDesk

ragulate query -n chunk_size_1000_k_5 -s experiment_chunk_size_and_k.py -m query_pipeline \
--var-name chunk_size --var-value 1000 --var-name k --var-value 5 --dataset BraintrustCodaHelpDesk
```

### Compare Recipes Example

This command should work:
```
ragulate compare -r chunk_size_500_k_2 -r chunk_size_1000_k_2 -r chunk_size_500_k_5 -r chunk_size_1000_k_5
```
