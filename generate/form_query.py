from tqdm import tqdm
from typing import List

from common.const import IDX_TO_LETTER
from common.model_utils import ModelFamily


def form_query(user_prompt: str, model_family: ModelFamily, system_prompt: str = None):
    if model_family == ModelFamily.OPENAI:
        return [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    elif model_family == ModelFamily.GEMINI:
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
    question_key: str,
    model_family: ModelFamily,
    sample_size: int = None,
    reasoning: bool = False,
):
    query_list = []
    inst_file = "multi_choice_query_reasoning" if reasoning else "multi_choice_query"
    with open(f"./instruction/{inst_file}.txt", encoding="utf-8") as f:
        instruction_template = "".join(f.readlines())

    for data in tqdm(data_list, desc="Forming Multichoice Queries..."):
        _sample_size = sample_size if sample_size else len(data["sampled_idxs_list"])

        for i in range(_sample_size):
            prompt = instruction_template.format(
                question=data["question"][question_key],
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


def build_finetune_content(text: str) -> list:
    return [{"type": "text", "text": text}]
