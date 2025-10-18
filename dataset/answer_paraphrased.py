import argparse
import random

from common.const import Dataset
from common.model_utils import get_model_family, PARAPHRASE_MODEL_MAP
from common.json_utils import load_from_json, dump_to_json
from common.random_utils import get_unique_permutations, get_seed
from common.slack_utils import slack_notify
from generate.form_query import form_multichoice_queries
from generate.parse_response import extract_multichoice_response, is_response_correct
from generate.open_src import batch_query_open_src


DATA_SELECTION_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    random.seed(get_seed())

    data_list = load_output(args)
    if any("sampled_idxs_list" not in data for data in data_list):
        add_sampled_idxs(data_list, args)

    for question_key in ["original"] + list(PARAPHRASE_MODEL_MAP.keys()):
        generate_answers(question_key, args)

    slack_notify(
        "Finished generating answers to paraphrased questions",
        model=DATA_SELECTION_MODEL_NAME,
        dataset=args.dataset.value,
    )


def generate_answers(question_key: str, args):
    print(f"Generate answer to questions paraphrased by: {question_key}")
    data_list = load_output(args)
    query_list = form_multichoice_queries(
        data_list,
        question_key,
        get_model_family(DATA_SELECTION_MODEL_NAME),
        args.sample_size,
    )

    responses = batch_query_open_src(
        query_list, DATA_SELECTION_MODEL_NAME, batch_size=args.sample_size
    )

    for i, data in enumerate(data_list):
        if "responses" not in data:
            data["responses"] = {}

        _responses = responses[(i * args.sample_size) : ((i + 1) * args.sample_size)]
        data["responses"][question_key] = _responses

        if "extracted_responses" not in data:
            data["extracted_responses"] = {}

        _extracted_responses = [
            extract_multichoice_response(res, len(data["options"]))
            for res in _responses
        ]
        data["extracted_responses"][question_key] = _extracted_responses

        if "responses_correct" not in data:
            data["responses_correct"] = {}

        data["responses_correct"][question_key] = [
            is_response_correct(res, data, sample_num)
            for sample_num, res in enumerate(_extracted_responses)
        ]

    save_output(data_list, args)


def add_sampled_idxs(data_list: list, args):
    for data in data_list:
        data["sampled_idxs_list"] = get_unique_permutations(
            len(data["options"]), args.sample_size
        )

    save_output(data_list, args)


def output_filepath(args) -> str:
    return f"./output/dataset/{args.dataset.value}/paraphrased.json"


def load_output(args):
    return load_from_json(output_filepath(args))


def save_output(data_list, args):
    dump_to_json(output_filepath(args), data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=Dataset)
    parser.add_argument("--sample-size", type=int, default=8)

    args = parser.parse_args()
    main(args)
