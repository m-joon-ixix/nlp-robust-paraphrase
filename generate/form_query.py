from tqdm import tqdm
from typing import List

from common.const import IDX_TO_LETTER
from common.model_utils import ModelFamily


def form_query(text: str, model_family: ModelFamily):
    if model_family == ModelFamily.OPENAI:
        message = {"role": "user", "content": [{"type": "text", "text": text}]}
    elif model_family == ModelFamily.GEMINI:
        message = {"role": "user", "parts": [{"text": text}]}
    else:
        # Open Source
        message = {"role": "user", "content": [{"type": "text", "text": text}]}

    return [message]


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


def build_finetune_content(text: str) -> list:
    return [{"type": "text", "text": text}]
