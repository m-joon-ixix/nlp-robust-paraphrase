from tqdm import tqdm
from typing import List

from common.const import IDX_TO_LETTER
from common.model_utils import ModelFamily
from common.string_utils import load_instruction


def form_query(user_prompt: str, model_family: ModelFamily, system_prompt: str = None):
    if model_family == ModelFamily.GEMINI:
        return [{"role": "user", "parts": [{"text": user_prompt}]}]
    elif model_family == ModelFamily.SNOWFLAKE:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # Open Source
        return [{"role": "user", "content": user_prompt}]


def form_multichoice_queries(
    data_list: List[dict],
    question_key: str | int,
    model_family: ModelFamily,
    sample_size: int = None,
    reasoning: bool = False,
    paraphrase_aware: bool = False,
):
    instruction_template = load_instruction(
        multichoice_query_inst_filename(
            reasoning=reasoning, paraphrase_aware=paraphrase_aware
        )
    )

    query_list = []
    for data in tqdm(data_list, desc="Forming Multichoice Queries..."):
        _sample_size = sample_size if sample_size else len(data["sampled_idxs_list"])
        # NOTE: the attr name differs before & after constructing HF dataset
        question_attr = "questions" if "questions" in data else "question"

        for i in range(_sample_size):
            prompt = instruction_template.format(
                question=data[question_attr][question_key],
                choices="\n".join(
                    [
                        form_single_option(data["options"], option_idx, letter_idx)
                        for letter_idx, option_idx in enumerate(
                            data["sampled_idxs_list"][i]
                        )
                    ]
                ),
            )

            query_list.append(form_query(prompt, model_family))

    print(f"{len(query_list)} queries formed from {len(data_list)} data examples.")
    return query_list


def form_single_option(
    options: List[str], option_idx: int, letter_idx: int = None
) -> str:
    if letter_idx is None:
        letter_idx = option_idx

    # e.g., "B. Clayton Kershaw"
    return IDX_TO_LETTER[letter_idx] + ". " + options[option_idx]


def multichoice_query_inst_filename(reasoning: bool, paraphrase_aware: bool) -> str:
    filename = "multi_choice_query"
    if reasoning:
        filename = filename + "_reasoning"

    if paraphrase_aware:
        filename = "paraphrase_aware_" + filename

    return filename
