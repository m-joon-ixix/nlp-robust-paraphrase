import argparse

from common.const import Dataset
from common.json_utils import load_from_json, dump_to_json
from common.model_utils import PARAPHRASE_MODEL_MAP


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    data_list = load_from_json(
        f"./output/dataset/{args.dataset.value}/paraphrased.json"
    )
    selected_data_list = [
        reduce_data(data) for data in data_list if is_data_to_be_selected(data)
    ]

    print(f"{len(selected_data_list)} examples out of {len(data_list)} selected.")

    dump_to_json(
        f"./output/dataset/{args.dataset.value}/selected.json", selected_data_list
    )


def is_data_to_be_selected(data: dict) -> bool:
    # sampled responses were all correct in 1 or 2 question types (perfectly correct in some questions, but not in other ones)
    num_of_all_correct_qtypes = 0
    for k in ["original"] + list(PARAPHRASE_MODEL_MAP.keys()):
        assert len(data["responses_correct"][k]) == 8
        if all(data["responses_correct"][k]):
            num_of_all_correct_qtypes += 1

    return 1 <= num_of_all_correct_qtypes <= 2


def reduce_data(data: dict) -> dict:
    # to save storage space, do not save raw responses to selected data file
    return {k: v for k, v in data.items() if k != "responses"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=Dataset)

    args = parser.parse_args()
    main(args)
