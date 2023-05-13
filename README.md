# Modeling and explaining beacons in code comprehension

## Setup
```
conda create -n program-comprehension python=3.8.11
conda env update --file env.yml --prune
conda activate program-comprehension
```
or
```
PIP_EXISTS_ACTION=w conda env create -f env.yml
```

## Data
Programs used in the behavioral experiments were sourced from the following repositories:
```
https://github.com/githubhuyang/refactory
https://github.com/jkoppel/QuixBugs
```

## To run

### Step 1

First, get model output information for each stimuli:

#### Mode 1: Get last-layer model activations for each input token  
For each problem, this mode generates a torch pkl containing a dict: tokens -> tensor.  
Path: `./experiments/custom-anonym`
```
python comprehend/model_outputs.py \
--model_names santa-coder \
--number_of_records -1 \
--dataset_name custom-anonym \
--dataset_path ./data \
--infer_interval 1 \
--expt_dir ./experiments \
--mode 1
```

#### Mode 2: Get model LL support sizes for each input token  
This mode generates a CSV for each problem.  
Path: `./experiments/custom-anonym`
```
python comprehend/model_outputs.py \
--model_names santa-coder \
--number_of_records -1 \
--dataset_name custom-anonym \
--dataset_path ./data \
--infer_interval 1 \
--expt_dir ./experiments \
--mode 2
```

### Step 2

Next, align model output data with participant responses available as Qualtrics data (which needs to be placed in `./data`)
```
python comprehend/prepare_dataset.py \
--responses_path data/code-comprehend_March 13, 2023_10.00.xlsx \ 
--token_wise_ll_support_path experiments/custom-anonym \
--token_wise_representations_path experiments/custom-anonym \
--out_path experiments/results
```

### Step 3
Analyze the prepared data by training models
```
python comprehend/analyze.py \
--dataset_path experiments/results \

```

## Test config

For `comprehend/model_outputs.py`

```
[
"--model_names", "codeberta-small",
"--number_of_records", "-1",
"--infer_interval", "2",
"--dataset_name", "custom-anonym",
"--dataset_path", "./data",
]
