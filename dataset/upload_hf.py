import pandas as pd
import random
import torch
from torch.utils.data import random_split
from time import time
from typing import List, Dict
from datasets import Dataset, DatasetDict

from common.json_utils import load_from_json
from common.secret_utils import load_secret
from common.const import Dataset as MyDataset, SPLITS, HF_DATASET_REPO_ID
from common.random_utils import get_seed

SUBSET_TO_DATASETS = {
    "general-knowledge": [MyDataset.MMLU, MyDataset.ARC, MyDataset.COMMONSENSE_QA],
    "math-reasoning": [MyDataset.MATH_QA],
}


def main():
    for subset, datasets in SUBSET_TO_DATASETS.items():
        split_to_data_list = {split: [] for split in SPLITS}

        for dataset in datasets:
            data_list = load_from_json(
                f"./output/dataset/{dataset.value}/selected.json"
            )
            for split, _data_list in split_data(data_list).items():
                split_to_data_list[split] += _data_list

        split_to_dataset = {
            split: convert_to_dataset(_data_list)
            for split, _data_list in split_to_data_list.items()
        }

        print_dataset_info(subset, split_to_dataset)
        upload_to_hf(DatasetDict(split_to_dataset), subset)


def print_dataset_info(subset: str, split_to_dataset: Dict[str, Dataset]):
    print(
        f"Subset '{subset}' Dataset Info -",
        ", ".join(
            [f"{split}: {len(dataset)}" for split, dataset in split_to_dataset.items()]
        ),
    )


def upload_to_hf(dataset_dict: DatasetDict, subset: str):
    dataset_dict.push_to_hub(
        HF_DATASET_REPO_ID, token=load_secret("hf_key"), config_name=subset
    )
    print(f"Finished uploading subset '{subset}' to HuggingFace!")


def split_data(data_list: List[dict]) -> Dict[str, List[dict]]:
    train_dataset_size = round(len(data_list) * 0.7)
    val_dataset_size = round(len(data_list) * 0.15)

    torch.manual_seed(get_seed())
    torch.cuda.manual_seed(get_seed())

    train_dataset, val_dataset, test_dataset = random_split(
        data_list,
        [
            train_dataset_size,
            val_dataset_size,
            len(data_list) - train_dataset_size - val_dataset_size,
        ],
    )

    return {
        "train": list(train_dataset),
        "validation": list(val_dataset),
        "test": list(test_dataset),
    }


def convert_to_dataset(data_list: List[dict]) -> Dataset:
    random.seed(get_seed())
    random.shuffle(data_list)

    df = pd.DataFrame(data_list)

    # change the column so that only a list of questions exist (remove the name of model that paraphrased)
    df["question"] = df["question"].apply(lambda d: list(d.values()))

    # rename columns
    df.rename(
        columns={"dataset": "source_dataset", "question": "questions"}, inplace=True
    )

    # drop unnecessary columns
    df.drop(columns=["extracted_responses", "responses_correct"], inplace=True)

    return Dataset.from_pandas(df)


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"Total run time : {end_time - start_time} sec.")
