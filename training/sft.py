import argparse
from time import time
from typing import Dict
from datetime import timedelta
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from huggingface_hub import login

from common.const import SUBSETS
from common.model_utils import model_name_to_dirname, models_to_finetune
from common.json_utils import load_from_json, dump_to_json
from common.yaml_utils import load_from_yaml
from common.slack_utils import slack_notify
from common.secret_utils import load_secret
from generate.open_src import get_model, get_tokenizer
from training.construct_data import training_data_filepath


def finetune(args):
    print("Running with the following arguments:")
    print(vars(args))

    login(token=load_secret("hf_key"))

    model = get_model(args.model_name)
    tokenizer = get_tokenizer(args.model_name)

    # Training arguments
    train_dataset = _get_finetune_dataset(args, "train")
    training_args = _get_training_args(args.model_name, args.subset, len(train_dataset))
    lora_config = LoraConfig(**load_from_yaml("./training/config/lora.yaml"))

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=_get_finetune_dataset(args, "validation"),
        formatting_func=build_formatting_fn(tokenizer),
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train the model
    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)

    # Manually save logs
    dump_to_json(
        f"{training_args.logging_dir}/log_history.json", trainer.state.log_history
    )

    # save training args in JSON format (by default, it only saves it into `.bin` format so it can only be read by `torch.load()`)
    save_training_args_to_json(training_args, lora_config)

    # Write down a description of this fine-tuning in the output_dir, if given
    if args.desc:
        _write_description(args.desc, training_args.output_dir)

    slack_notify("Fine-tuning Completed!", output_dir=training_args.output_dir)


def _get_training_args(
    model_name: str, subset: str, training_set_size: int
) -> TrainingArguments:
    training_config = load_from_yaml("./training/config/sft.yaml")

    # auto-compute `num_train_epochs`
    num_of_epochs = round(
        (training_config.pop("total_train_examples") + 1) / training_set_size
    )
    training_config["num_train_epochs"] = num_of_epochs
    print(f"Training Set Size: {training_set_size}, Training Epochs: {num_of_epochs}")

    output_dir = f"./training_output/{subset}/{model_name_to_dirname(model_name)}/sft"

    return SFTConfig(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        **training_config,
    )


def _get_finetune_dataset(args, split: str) -> Dataset:
    return Dataset.from_list(
        load_from_json(training_data_filepath(args.subset, args.model_name, split))
    )


def build_formatting_fn(tokenizer):
    # Formatting function: converts a chat-style example into a single string using the model's chat template.
    # Expected example schema:
    #   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    def _format_example(example: Dict[str, list]) -> str:
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    return _format_example


def save_training_args_to_json(
    training_args: TrainingArguments, lora_config: LoraConfig
):
    training_args_dict = training_args.to_dict()
    training_args_dict["lora_config"] = lora_config.to_dict()

    dump_to_json(f"{training_args.output_dir}/training_args.json", training_args_dict)


def _write_description(description: str, output_dir: str):
    output_path = f"{output_dir}/description.txt"
    with open(output_path, "w") as f:
        f.write(description)

    print(f"Wrote down description to: {output_path}")


# CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python training/sft.py --model-name meta-llama/Llama-3.1-8B-Instruct --subset general-knowledge --desc "Paraphrase-aware SFT"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, choices=models_to_finetune())
    parser.add_argument("--subset", type=str, choices=SUBSETS)
    parser.add_argument("--desc", type=str, required=False)

    args = parser.parse_args()
    start_time = time()

    finetune(args)

    time_formatted = str(timedelta(seconds=(time() - start_time)))
    print(f"Training Completed (Time Elapsed: {time_formatted})")
