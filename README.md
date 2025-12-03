# RoParQ: Paraphrase-Aware Alignment of Large Language Models Towards Robustness to Paraphrased Questions

## Contributions of this Project
(1) Constructing the **RoParQ** Benchmark<br>
(2) Proposing the **XParaCon** Metric<br>
(3) Demonstrating the Efficacy of **Paraphrase-Aware Alignment**


## Installing Dependencies
> Python Version: 3.12.11

Install the required dependencies using the following command.
```sh
pip install -r requirements.txt
```

## Managing Secrets
This project manages confidential information in a secret file that is gitignored.<br>
A file named `secrets.yaml` must exist in the root of this repository.<br>
This file must contain the following contents to enable all functionalities.
```yaml
slack_webhook_url: {your_webhook_url}
hf_key: {your_huggingface_key}
api_keys:
    gemini:
        - {api_key1}
        - {api_key2}
        ...
snowflake:
    account: {snowflake_account}
    user: {snowflake_user}
    password: {snowflake_password}
    role: {snowflake_role}
    database: {snowflake_database}
    schema: {snowflake_schema}
    warehouse: {snowflake_warehouse}
```

## Run Manuals

### 1. Constructing the RoParQ dataset

#### (1) Load Raw Dataset from [HuggingFace Unified-MCQA](https://huggingface.co/datasets/pszemraj/unified-mcqa)
> Dataset Preprocessing Included
```sh
PYTHONPATH=. python dataset/save_raw_dataset.py
```

#### (2) Paraphrase Questions with Proprietary Models
```sh
PYTHONPATH=. python dataset/paraphrase_questions.py --dataset {dataset} --model-key {model}
# PYTHONPATH=. python dataset/paraphrase_questions.py --dataset mmlu --model-key gemini
```

#### (3) Generate Answers with Judge Model to Check Consistency of Each Example
```sh
PYTHONPATH=. python dataset/answer_paraphrased.py --dataset {dataset}
# PYTHONPATH=. python dataset/answer_paraphrased.py --dataset mmlu
```

#### (4) Filter Examples that Exhibit "Inconsistent Confidence"
```sh
PYTHONPATH=. python dataset/select_data.py --dataset {dataset}
# PYTHONPATH=. python dataset/select_data.py --dataset mmlu
```

#### (5) Upload the Dataset RoParQ to [HuggingFace](https://huggingface.co/datasets/m-joon-ixix/RoParQ), and Save in Local Environment as JSON
```sh
PYTHONPATH=. python dataset/upload_hf.py
PYTHONPATH=. python dataset/load_from_hf.py
```

### 2. Generating Answers to the RoParQ dataset

#### To Generate Answers for a Single (Subset, Model, Split) Setting
```sh
CUDA_VISIBLE_DEVICES={device} PYTHONPATH=. python run/generate_responses.py --subset {subset} --model-name {model} --split {split}
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python run/generate_responses.py --subset general-knowledge --model-name meta-llama/Llama-3.1-8B-Instruct --split test
```

#### To Generate Answers for All Settings at Once
> This will take very long. Not recommended in real-world settings.
```sh
# Assuming that GPU devices 0, 1 are in use
sh run/generate_responses_all.sh "0,1"
```

#### To Check Progress of Generating Answers
- Run `run/check_progress.ipynb` (Jupyter Notebook)

### 3. Paraphrase-Aware Fine-tuning

#### (1) Construct Training Data (training set & validation set)
```sh
PYTHONPATH=. python training/construct_data.py --subset {subset} --model-name {model}
# PYTHONPATH=. python training/construct_data.py --subset general-knowledge --model-name meta-llama/Llama-3.1-8B-Instruct
```

#### (2) Run Fine-tuning
- Hyperparameters for Supervised Fine-Tuning (SFT) and LoRA are managed in these yaml files.
    - `training/config/sft.yaml`
    - `training/config/lora.yaml`

- For a single (Model, Subset) setting
```sh
CUDA_VISIBLE_DEVICES={device} PYTHONPATH=. python training/sft.py --model-name {model} --subset {subset}
# CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python training/sft.py --model-name meta-llama/Llama-3.1-8B-Instruct --subset general-knowledge
```

- For all settings at once
```sh
# Assuming that GPU devices 0, 1 are in use
sh training/sft_all.sh "0,1"
```

### 4. Evaluation

#### Compute and Save Metrics into Pickle (.pkl) Files
```sh
# For all open-source (pre-trained), proprietary models
PYTHONPATH=. python evaluate/run_eval.py --split test --model-ver base

# For fine-tuned models
PYTHONPATH=. python evaluate/run_eval.py --split test --model-ver sft
```

#### View Stats
- Run `evaluate/view_stats.ipynb` (Jupyter Notebook)

#### Visualize Results into Plots
- Run `evaluate/plot.ipynb` (Jupyter Notebook)
