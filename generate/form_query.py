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
        return [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]


def form_multichoice_queries(data_list: List[dict], model_family: ModelFamily):
    query_list = []
    with open(f"./instruction/multi_choice_query.txt", encoding="utf-8") as f:
        instruction = "".join(f.readlines())

    for data in tqdm(data_list, desc="Forming Multichoice Queries..."):
        query = "### Question:\n" + data["question"] + "\n### Options:\n"
        for i in range(len(data["options"])):
            query += f"{IDX_TO_LETTER[i]}. {data['options'][i]}\n"

        query += instruction
        query_list.append(form_query(query, model_family))

    return query_list


def form_single_option(options: List[str], idx: int) -> str:
    # e.g., "B. Clayton Kershaw"
    return IDX_TO_LETTER[idx] + ". " + options[idx]


def build_finetune_content(text: str) -> list:
    return [{"type": "text", "text": text}]
