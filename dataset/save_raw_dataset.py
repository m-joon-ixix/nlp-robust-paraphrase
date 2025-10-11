import uuid
from datasets import load_dataset, concatenate_datasets, Dataset

from common.const import DATASET_CACHE_DIR
from common.json_utils import dump_to_json

HF_DATASET_ID = "pszemraj/unified-mcqa"

N_CHOICE_TO_DATASETS = {
    4: ["mmlu", "arc_easy", "arc_challenge"],
    5: ["math_qa", "commonsense_qa"],
}


def main():
    n_choice_to_dataset = {
        n_choice: get_filtered_dataset(n_choice) for n_choice in [4, 5]
    }

    data_list = concatenate_datasets(list(n_choice_to_dataset.values())).to_list()
    uuid_list = get_uuids(len(data_list))

    formatted_data_list = [
        format_data(data, uuid) for data, uuid in zip(data_list, uuid_list)
    ]

    dump_to_json("./output/dataset/origin.json", formatted_data_list)


def get_filtered_dataset(n_choice: int) -> Dataset:
    raw_dataset = load_dataset(
        HF_DATASET_ID, f"{n_choice}-choice", cache_dir=DATASET_CACHE_DIR
    )["train"]
    print(f"Loaded {n_choice}-choice data from {HF_DATASET_ID}")

    dataset = raw_dataset.filter(lambda example: len(example["context"]) == 0)
    dataset = dataset.filter(lambda d: "_" not in d["question"])
    dataset = dataset.filter(lambda d: d["question"].endswith("?"))
    dataset = dataset.filter(
        lambda d: d["source_dataset"] in N_CHOICE_TO_DATASETS[n_choice]
    )
    dataset = dataset.filter(
        lambda d: len(d["question"].split(".")) <= 3
        and len(d["question"].split(" ")) >= 10
    )

    return dataset


def get_uuids(n: int) -> list:
    uuid_set = set()
    while len(uuid_set) < n:
        uuid_set.add(uuid.uuid4().hex)

    return list(uuid_set)


def format_data(data, uuid):
    return {
        "id": uuid,
        "dataset": (
            "arc"
            if data["source_dataset"].startswith("arc")
            else data["source_dataset"]
        ),
        "question": data["question"],
        "options": data["choices"],
        "answer_idx": data["label"],
    }


if __name__ == "__main__":
    main()
