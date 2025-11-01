import argparse
import random
from typing import List

from common.const import SUBSETS, SPLITS
from common.model_utils import model_name_to_dirname, models_to_finetune
from common.json_utils import load_from_json, dump_to_json
from common.random_utils import get_seed
from generate.form_query import form_single_option

METRICS = ["accuracy", "xparacon"]


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    subsets = SUBSETS if args.subset is None else [args.subset]
    model_names = models_to_finetune() if args.model_name is None else [args.model_name]

    random.seed(get_seed())
    for subset in subsets:
        for model_name in model_names:
            for split in SPLITS[:2]:  # train, validation
                construct(subset, model_name, split)


def construct(subset: str, model_name: str, split: str):
    response_data_list = load_from_json(response_filepath(subset, model_name, split))

    inst_file = "paraphrase_aware_multi_choice_query"
    if "reasoning" in subset:
        inst_file += "_reasoning"

    with open(f"./instruction/{inst_file}.txt", encoding="utf-8") as f:
        inst_template = "".join(f.readlines())

    data_list = []
    for data in response_data_list:
        sample_nums = sample_numbers_for_each_q(data)
        if any(e is None for e in sample_nums):
            continue  # if any of the questions had no correct response, remove this example

        for q_idx, sample_num in enumerate(sample_nums):
            user_query = build_user_query(data, q_idx, sample_num, inst_template)
            assistant_response = build_assistant_response(data, q_idx, sample_num)
            formatted_data = {
                "messages": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

            data_list.append(formatted_data)

    random.shuffle(data_list)
    print(f"Original Data: {len(response_data_list)}, Training Data: {len(data_list)}")
    dump_to_json(training_data_filepath(subset, model_name, split), data_list)


def sample_numbers_for_each_q(data: dict) -> List[int]:
    sample_nums = []
    for i in range(3):  # for each question
        _correct_sample_nums = [
            n for n, is_corr in enumerate(data["responses_correct"][i]) if is_corr
        ]
        sample_nums.append(
            random.choice(_correct_sample_nums)
            if len(_correct_sample_nums) > 0
            else None
        )

    assert len(sample_nums) == 3
    return sample_nums


def build_user_query(
    data: dict, question_idx: int, sample_num: int, instruction_template: str
) -> str:
    return instruction_template.format(
        question=data["questions"][question_idx],
        choices="\n".join(
            [
                form_single_option(data["options"], option_idx, letter_idx)
                for letter_idx, option_idx in enumerate(
                    data["sampled_idxs_list"][sample_num]
                )
            ]
        ),
    )


def build_assistant_response(data: dict, question_idx: int, sample_num: int) -> str:
    next_question_idx = (question_idx + 1) % 3
    return f"Paraphrase: {data['questions'][next_question_idx]}\n{data['responses'][question_idx][sample_num]}"


def response_filepath(subset: str, model_name: str, split: str) -> str:
    return f"./output/{subset}/response/{model_name_to_dirname(model_name)}/base/original_{split}.json"


def training_data_filepath(subset: str, model_name: str, split: str) -> str:
    return f"./output/{subset}/training_data/{model_name_to_dirname(model_name)}/{split}.json"


# ex. PYTHONPATH=. python training/construct_data.py --subset general-knowledge --model-name meta-llama/Llama-3.1-8B-Instruct
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subset", type=str, choices=SUBSETS, required=False)
    parser.add_argument(
        "--model-name", type=str, choices=models_to_finetune(), required=False
    )

    args = parser.parse_args()
    main(args)
