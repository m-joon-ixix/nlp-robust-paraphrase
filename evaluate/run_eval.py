import os
import argparse
import pandas as pd

from common.const import SUBSETS, SPLITS
from common.model_utils import (
    OPEN_SRC_MODELS,
    PROPRIETARY_MODELS,
    model_name_to_dirname,
    model_sort_key,
    models_to_finetune,
)
from common.json_utils import load_from_json
from common.pickle_utils import load_from_pickle, dump_to_pickle
from evaluate.utils import compute_accuracy, compute_xparacon

METRICS = ["accuracy", "xparacon"]


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    subsets = SUBSETS if args.subset is None else [args.subset]
    model_names = (
        _get_all_models_for_eval(args.model_ver)
        if args.model_name is None
        else [args.model_name]
    )
    splits = SPLITS if args.split is None else [args.split]

    for subset in subsets:
        for split in splits:
            stat_df = load_stat_df(subset, split)

            for model_name in model_names:
                df_index = stat_df_index_name(model_name, args.model_ver)
                metric_dict = compute_metrics(subset, model_name, split, args.model_ver)
                if metric_dict is None:  # generating not finished
                    continue

                for metric, value in metric_dict.items():
                    assert metric in METRICS
                    stat_df.loc[df_index, metric] = round(value, 6)

            sorted_index_list = sorted(list(stat_df.index), key=model_sort_key)
            stat_df = stat_df.reindex(sorted_index_list)

            dump_to_pickle(stats_filepath(subset, split), stat_df)


def compute_metrics(subset: str, model_name: str, split: str, model_ver: str) -> dict:
    filepath = response_filepath(subset, model_name, split, model_ver)
    if not os.path.exists(filepath):
        return None

    data_list = load_from_json(filepath)
    if any(validate_response_data(data) == False for data in data_list):
        return None

    return {
        "accuracy": compute_accuracy(data_list),
        "xparacon": compute_xparacon(data_list),
    }


def validate_response_data(data: dict) -> bool:
    if data.get("responses_correct") is None:
        return False

    if len(data["responses_correct"]) != 3:
        return False

    for i in range(3):
        if not isinstance(data["responses_correct"][i], list):
            return False

        if len(data["responses_correct"][i]) != 8:
            return False

    return True


def response_filepath(subset: str, model_name: str, split: str, model_ver: str) -> str:
    return f"./output/{subset}/response/{model_name_to_dirname(model_name)}/{model_ver}/original_{split}.json"


def stats_filepath(subset: str, split: str) -> str:
    return f"./output/{subset}/stat/original_{split}.pkl"


def load_stat_df(subset: str, split: str) -> str:
    filepath = stats_filepath(subset, split)
    if os.path.exists(filepath):
        return load_from_pickle(filepath)
    else:
        return pd.DataFrame(columns=METRICS)


def stat_df_index_name(model_name: str, model_ver: str) -> str:
    model_name = model_name.split("/")[-1]
    if model_ver == "base":
        return model_name
    else:
        return model_name + "_" + model_ver


def _get_all_models_for_eval(model_ver: str) -> list:
    if model_ver == "sft":
        return models_to_finetune()
    else:
        return OPEN_SRC_MODELS + PROPRIETARY_MODELS


# ex. PYTHONPATH=. python evaluate/run_eval.py --subset general-knowledge --model-name meta-llama/Llama-3.1-8B-Instruct --split test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subset", type=str, choices=SUBSETS, required=False)
    parser.add_argument(
        "--model-name",
        type=str,
        choices=OPEN_SRC_MODELS + PROPRIETARY_MODELS,
        required=False,
    )
    parser.add_argument("--split", type=str, choices=SPLITS, required=False)
    parser.add_argument(
        "--model-ver",
        type=str,
        default="base",
        help="Model version (base, sft, etc.)",
    )

    args = parser.parse_args()
    main(args)
