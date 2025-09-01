
## Library Overview

This implementation includes the following models:

## Installation

First, create a python 3.7 environment and install dependencies:

```bash
virtualenv -p python3.7 hyp_kg_env
source hyp_kg_env/bin/activate
pip install -r requirements.txt
```

Then, set environment variables and activate your environment:

```bash
source set_env.sh
```

If you are in anaconda, you can create a new environment and install the dependencies:

```bash
conda create -n hyp_kg_env python=3.7 anaconda
conda activate hyp_kg_env
pip install -r requirements.txt
./set_env.sh
```

## Datasets

Download and pre-process the datasets:

```bash
source datasets/download.sh
python datasets/process.py
```

If installed via anaconda then run following commands:

```bash
./datasets/download.sh
python datasets/process1.py
```

## Note
Please provide appropriate cuda version in the example scripts located in the examples folder.

## Usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding
