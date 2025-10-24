from time import time
from datasets import load_dataset

from common.const import HF_DATASET_REPO_ID, DATASET_CACHE_DIR
from common.json_utils import dump_to_json
from dataset.upload_hf import SUBSET_TO_DATASETS


def main():
    for subset in SUBSET_TO_DATASETS.keys():
        dataset_dict = load_dataset(
            HF_DATASET_REPO_ID, subset, cache_dir=DATASET_CACHE_DIR
        )

        train_data_list = dataset_dict["train"].to_list()
        dump_to_json(f"./output/{subset}/dataset/train.json", train_data_list)

        test_data_list = (
            dataset_dict["validation"].to_list() + dataset_dict["test"].to_list()
        )
        dump_to_json(f"./output/{subset}/dataset/test.json", test_data_list)


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"Total run time : {end_time - start_time} sec.")
