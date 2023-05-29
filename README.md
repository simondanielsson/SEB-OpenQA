# SEB-workspace

This is a temporary workspace for a Master's thesis conducted for SEB.

### 1. The ORQA pipeline 

:checkered_flag: The goal of an open-retrieval question-answering (ORQA) system is to answer a given question
using a (potentially huge) collection of documents. Our pipeline is extractive, meaning answer 
is a minimal span containing the answer to the question, found in one of the documents. 


#### 1.1 Format 
The ORQA pipeline inputs a SQuAD-like ORQA dataset (like *Natural Questions*), 
and outputs results to `evaluation/output/<dataset_name>/<date_and_time>/`. The results 
include the predictions, experiment configurations, and metadata (such as inference runtime). 

Predictions conform to the format required by the SQuAD evaluation script: namely pairs of 
(`question_ids`, `answer_text`).

### 2. Installation

First create a virtual environment

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
. .env
```

:exclamation: *Note: If you want GPU acceleration, you additionally have to download the 
GPU-related dependencies to PyTorch and FAISS. Do this before installing the other dependencies
using `pip`. For instance, using conda,*

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

and 

```bash
pip install farm-haystack[all-gpu] 
```

### 3. How to run

Execute the pipeline by running 

```bash
python3 src/
```

This requires the required train/dev data to be saved to disk. See Section 
*4. Downloading data* below for more information.   

### 4. Configure experiments

There are two configurations files governing all ORQA experiments: 
an [experiment config](configs/config_open_domain.yaml) and a 
[pipeline config](configs/basic.haystack-pipeline.yml). By default, the inference 
pipeline assumes these are found in `configs/`. The pipeline config used by the experiment
is an entry in the experiment config (`pipeline_config`). 

There is also an option to supply different configs, instead of changing the default configs.
This is particularly useful for reproducing old experiments using the configs supplied by the pipeline 
output (found in `evaluation/output/<dataset_name>/<run_id>/`). See below for how this is done.

```
usage: __main__.py [-h] [--config_path [CONFIG_PATH]] [--pipeline_config_path [PIPELINE_CONFIG_PATH]]

Run inference on a ORQA pipeline

options:
  -h, --help            show this help message and exit
  --config_path [CONFIG_PATH]
                        Config for this run, defaults to /Users/simondanielsson/Documents/Master/SEB-
                        workspace/configs/config_open_domain.yaml.
  --pipeline_config_path [PIPELINE_CONFIG_PATH]
                        Pipeline config for this run, defaults to what is found in /Users/simondanielsson/Documents/Master/SEB-
                        workspace/configs/config_open_domain.yaml.
```

### 4. Downloading data

The pipeline currently only supports loading data from disk. Additionally, Haystack
is assuming all data is SQuAD-formatted. We provide scripts for converting a few datasets into
SQuAD-format, strongly influenced by e.g. Facebook's own Natural Questions simplification script. 

#### 4.1 Converting Natural Questions
1. Download the Natural Questions train and dev set from the [official website](https://ai.google.com/research/NaturalQuestions/download).
2. Invoke `squad_fmt_conversion/nq_to_squad/nq_to_squad.py` as described in the module docstring. The train set should 
be converted to JSON Lines using the `--as_jsonl` flag.
3. Point to the train and dev set paths in the config.

#### 4.2 Converting TriviaQA

Have a look in `squad_fmt_conversion/triviaQA_to_squad/README.md`.