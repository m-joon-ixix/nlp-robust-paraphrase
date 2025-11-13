import os
import argparse
from typing import List

from common.const import SUBSETS, SPLITS
from common.model_utils import (
    OPEN_SRC_MODELS,
    PROPRIETARY_MODELS,
    model_name_to_dirname,
    get_model_family,
)
from common.json_utils import load_from_json, dump_to_json
from generate.form_query import form_multichoice_queries
from generate.parse_response import extract_multichoice_response, is_response_correct
from generate.open_src import batch_query_open_src
from generate.api import batch_query_api, check_generate_failure
from training.sft import get_training_output_dir


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    if args.question_idx is None:
        for question_idx in [0, 1, 2]:
            generate_for_single_question(args, question_idx)
    else:
        generate_for_single_question(args, args.question_idx)


def generate_for_single_question(args, question_idx: int):
    print(">" * 10, f"Start generating responses for question {question_idx}", "<" * 10)
    data_list = load_data_list(args)
    sample_size = get_sample_size(data_list)

    model_family = get_model_family(args.model_name)
    query_list = form_multichoice_queries(
        data_list,
        question_idx,
        model_family,
        sample_size=sample_size,
        reasoning=("reasoning" in args.subset),
        paraphrase_aware=(args.model_ver == "sft"),
    )

    if model_family.is_open_src():
        responses = batch_query_open_src(
            query_list,
            args.model_name,
            batch_size=sample_size,
            peft_dir=(
                get_training_output_dir(args.subset, args.model_name)
                if args.model_ver == "sft"
                else None
            ),
        )
    else:
        responses = batch_query_api(query_list, args.model_name)
        check_generate_failure(responses, args.model_name, response_filepath(args))

    data_list = load_data_list(args)  # reload to fetch the latest file
    for i, data in enumerate(data_list):
        if "responses" not in data:
            data["responses"] = [None] * 3

        _responses = responses[(i * sample_size) : ((i + 1) * sample_size)]
        data["responses"][question_idx] = _responses

        if "extracted_responses" not in data:
            data["extracted_responses"] = [None] * 3

        _extracted_responses = [
            extract_multichoice_response(res, len(data["options"]))
            for res in _responses
        ]
        data["extracted_responses"][question_idx] = _extracted_responses

        if "responses_correct" not in data:
            data["responses_correct"] = [None] * 3

        data["responses_correct"][question_idx] = [
            is_response_correct(res, data, sample_num)
            for sample_num, res in enumerate(_extracted_responses)
        ]

    dump_to_json(response_filepath(args), data_list)


def load_data_list(args) -> List[dict]:
    if os.path.exists(response_filepath(args)):
        return load_from_json(response_filepath(args))
    else:
        return load_from_json(f"./output/{args.subset}/dataset/{args.split}.json")


def get_sample_size(data_list: List[dict]) -> int:
    sample_sizes = set()
    for data in data_list:
        sample_sizes.add(len(data["sampled_idxs_list"]))

    assert len(sample_sizes) == 1, "Sample size within a data_list should be consistent"
    return list(sample_sizes)[0]


def response_filepath(args) -> str:
    return f"./output/{args.subset}/response/{model_name_to_dirname(args.model_name)}/{args.model_ver}/original_{args.split}.json"


# ex. CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python run/generate_responses.py --subset general-knowledge --model-name meta-llama/Llama-3.1-8B-Instruct --split test --question-idx 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subset", type=str, choices=SUBSETS)
    parser.add_argument(
        "--model-name", type=str, choices=OPEN_SRC_MODELS + PROPRIETARY_MODELS
    )
    parser.add_argument("--split", type=str, choices=SPLITS)
    parser.add_argument("--question-idx", type=int, choices=[0, 1, 2], required=False)
    parser.add_argument(
        "--model-ver",
        type=str,
        default="base",
        help="Model version (base, sft, etc.)",
    )

    args = parser.parse_args()

    if args.model_ver == "sft":
        assert args.model_name in OPEN_SRC_MODELS, "SFT only in open source models"

    main(args)
