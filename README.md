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
Documentation TBD

### 2. Generating Answers to the RoParQ dataset
Documentation TBD

### 3. Paraphrase-Aware Fine-tuning
Documentation TBD

### 4. Evaluation
Documentation TBD
