import re
from typing import List


def load_instruction(instruction_name: str) -> str:
    with open(f"instruction/{instruction_name}.txt", encoding="utf-8") as f:
        return "".join(f.readlines())


def split_to_sentences(text: str) -> List[str]:
    # delimiter: punctuation followed by one or more whitespaces
    return re.split(r"(?<=[.!?])\s+", text)
