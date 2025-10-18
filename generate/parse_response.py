import re

from common.const import FAILED_TOKEN, IDX_TO_LETTER, LETTER_TO_IDX


def extract_multichoice_response(response: str, num_options: int):
    response = response.strip()
    response = response.split("Answer:")[-1].strip()
    if "Reasoning:" in response:
        response = response.split("Reasoning:")[0].strip()

    # pattern differs by the number of options. If there are 4 options, it looks for A-D
    # match a single uppercase letter that is not preceded by any letter/digit and followed by a dot
    pattern = re.compile(rf"(?<![A-Za-z0-9])([A-{IDX_TO_LETTER[num_options - 1]}])\.")
    match = pattern.search(response)
    if match:
        return LETTER_TO_IDX[match.group(1)]
    else:
        return FAILED_TOKEN


def is_response_correct(extracted_response, data: dict, sample_num: int) -> bool:
    if extracted_response == FAILED_TOKEN:
        return False

    return (
        data["sampled_idxs_list"][sample_num][extracted_response] == data["answer_idx"]
    )
