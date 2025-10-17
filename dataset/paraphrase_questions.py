import os
import argparse
from datetime import datetime
from time import sleep
from tqdm import tqdm

from generate.api import batch_query_api, check_generate_failure, get_api_keys
from generate.form_query import form_query, form_single_option
from common.const import Dataset, FAILED_TOKEN
from common.model_utils import (
    ModelFamily,
    get_model_family,
    PARAPHRASE_MODEL_MAP,
    MODEL_TO_RPD_LIMIT,
)
from common.string_utils import load_instruction
from common.json_utils import load_from_json, dump_to_json
from common.slack_utils import slack_notify

CHUNK_SIZE = 500


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    origin_data_list = load_from_json("./output/dataset/origin.json")
    origin_data_list = [
        data for data in origin_data_list if data["dataset"] == args.dataset.value
    ]

    if args.chunk_num:
        start_chunk_num = args.chunk_num
        last_chunk_num = args.chunk_num
    else:
        start_chunk_num = check_next_chunk(args)
        last_chunk_num = _get_last_chunk_num(len(origin_data_list))

    if start_chunk_num > last_chunk_num:
        print("Paraphrasing questions is finished. Terminating...")
        return

    model_name = PARAPHRASE_MODEL_MAP[args.model_key]
    print(
        f"Dataset: {args.dataset.value} (len: {len(origin_data_list)}), Model: {model_name}, Chunk Numbers to run: {start_chunk_num}-{last_chunk_num}"
    )

    api_usage_limit = daily_api_usage_limit(model_name)
    for chunk_num in range(start_chunk_num, last_chunk_num + 1):
        if get_api_usage_today(model_name) + CHUNK_SIZE > api_usage_limit:
            wait_until_tomorrow()

        filepath = output_filepath(args.dataset, chunk_num)
        if os.path.exists(filepath):
            data_list = load_from_json(filepath)
        else:
            print(f"{filepath} does not exist. Fetching data from origin file.")
            data_list = origin_data_list[
                (chunk_num * CHUNK_SIZE) : ((chunk_num + 1) * CHUNK_SIZE)
            ]

        run_chunk(data_list, args, filepath)

    merge_chunks(args.dataset)

    if args.chunk_num is None:
        slack_notify(
            "Finished paraphrasing questions.",
            dataset=args.dataset.value,
            model=model_name,
        )


def run_chunk(data_list, args, filepath):
    model_name = PARAPHRASE_MODEL_MAP[args.model_key]

    # NOTE: only query API for data examples that do not have the paraphrased question
    original_idxs = [
        i
        for i, data in enumerate(data_list)
        if not _paraphrased_question_exists(data, args.model_key)
    ]

    query_list = build_query_list(
        [data_list[i] for i in original_idxs], get_model_family(model_name)
    )
    responses = batch_query_api(query_list, model_name)
    increment_api_usage_today(model_name, len(query_list))

    check_generate_failure(
        responses, model_name, filepath, required_str="New Question:"
    )

    for i, raw_response in zip(original_idxs, responses):
        paraphrased_question = raw_response.split("New Question:")[-1].strip()
        data_list[i]["question"][args.model_key] = paraphrased_question

    dump_to_json(filepath, data_list)


def merge_chunks(dataset: Dataset):
    dataset_size = sum(
        data["dataset"] == dataset.value
        for data in load_from_json("./output/dataset/origin.json")
    )

    data_list = []
    for chunk_num in tqdm(
        range(_get_last_chunk_num(dataset_size) + 1), desc=f"Merging chunks"
    ):
        filepath = output_filepath(dataset, chunk_num)
        if not os.path.exists(filepath):
            print(f"{filepath} does not exist. Stop merging chunks.")
            return

        _data_list = [
            data
            for data in load_from_json(filepath, print_msg=False)
            if not _paraphrase_refusal_exists(data)
        ]
        data_list.extend(_data_list)

    print(f"{dataset.value} merged (before: {dataset_size}, after: {len(data_list)})")
    dump_to_json(f"./output/dataset/{dataset.value}/paraphrased.json", data_list)


def _paraphrased_question_exists(data: dict, model_key: str) -> bool:
    result = data["question"].get(model_key)
    return result is not None and len(result) > 0 and result != FAILED_TOKEN


def _paraphrase_refusal_exists(data: dict) -> bool:
    for para_question in data["question"].values():
        if para_question.startswith("I apologize, but I cannot assist with "):
            return True

    return False


def build_query_list(data_list: list, model_family: ModelFamily) -> list:
    instruction_template = load_instruction("paraphrase_question")
    system_prompt = (
        load_instruction("paraphrase_question_system")
        if model_family == ModelFamily.SNOWFLAKE
        else None
    )

    query_list = []
    for data in tqdm(data_list, desc="Building queries to paraphrase questions"):
        instruction = instruction_template.format(
            question=data["question"]["original"],
            choices="\n".join(
                [
                    form_single_option(data["options"], i)
                    for i in range(len(data["options"]))
                ]
            ),
            answer=form_single_option(data["options"], data["answer_idx"]),
        )

        query_list.append(
            form_query(instruction, model_family, system_prompt=system_prompt)
        )

    return query_list


def check_next_chunk(args) -> int:
    """
    Returns:
        the next chunk number (zero-based idx) to paraphrase questions on
    """
    next_chunk = 0
    while True:
        filepath = output_filepath(args.dataset, next_chunk)
        if not os.path.exists(filepath):
            return next_chunk

        if (
            args.model_key
            not in load_from_json(filepath, print_msg=False)[-1]["question"]
        ):
            return next_chunk

        next_chunk += 1


def _get_last_chunk_num(dataset_size: int) -> int:
    return int((dataset_size - 1) / CHUNK_SIZE)  # inclusive


def output_filepath(dataset: Dataset, chunk_num: int) -> str:
    assert chunk_num < 100
    return f"./output/dataset/{dataset.value}/paraphrased/chunk{chunk_num:02d}.json"


def daily_api_usage_limit(model_name: str) -> int:
    model_family = get_model_family(model_name)

    limit = MODEL_TO_RPD_LIMIT[model_name]
    if model_family != ModelFamily.SNOWFLAKE:
        limit *= len(get_api_keys(model_family))

    # 95% of the official RPD used as limit, to leave some space for unexpected extra requests
    return int(limit * 0.95)


def get_api_usage_today(model_name: str) -> int:
    filepath = "./output/api_usage.json"
    if os.path.exists(filepath):
        api_usage = load_from_json(filepath)
    else:
        api_usage = {}

    return api_usage.get(model_name, {}).get(strftoday(), 0)


def increment_api_usage_today(model_name: str, count: int):
    filepath = "./output/api_usage.json"
    if os.path.exists(filepath):
        api_usage = load_from_json(filepath)
    else:
        api_usage = {}

    if model_name not in api_usage:
        api_usage[model_name] = {}

    prev_count = api_usage[model_name].get(strftoday(), 0)
    api_usage[model_name][strftoday()] = prev_count + count

    dump_to_json(filepath, api_usage)


def wait_until_tomorrow():
    print(
        "Daily API request limit expected to exceed. Waiting until tomorrow.",
        end="",
        flush=True,
    )

    today = strftoday()
    while strftoday() == today:
        sleep(60 * 20)  # 20 mins
        print(".", end="", flush=True)

    print("\nWaiting done. Resuming API calls.")


def strftoday() -> str:
    return datetime.now().strftime("%m-%d-%Y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=Dataset)
    parser.add_argument(
        "--model-key", type=str, choices=list(PARAPHRASE_MODEL_MAP.keys())
    )
    parser.add_argument(
        "--chunk-num",
        type=int,
        required=False,
        help="if given, only runs on that chunk",
    )

    args = parser.parse_args()
    main(args)
