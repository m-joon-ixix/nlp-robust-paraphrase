import os
from enum import Enum

## Paths
DATASET_CACHE_DIR = os.path.expanduser("~/hf/datasets")
MODEL_CACHE_DIR = os.path.expanduser("~/hf/hub")

## Constants
HF_DATASET_REPO_ID = "m-joon-ixix/RoParQ"
SUBSETS = ["general-knowledge", "math-reasoning"]
SPLITS = ["train", "test"]


## Enums
class Dataset(Enum):
    MMLU = "mmlu"
    ARC = "arc"
    COMMONSENSE_QA = "commonsense_qa"
    MATH_QA = "math_qa"

    def is_reasoning_necessary(self) -> bool:
        return self in [Dataset.MATH_QA]


## Indexing
IDX_TO_LETTER = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

LETTER_TO_IDX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
}

FLAG_TO_SPLIT = {
    "t": "train",
    "v": "val",
    "e": "test",
}

## Custom Tokens
FAILED_TOKEN = "<none>"
PROHIBITED_CONTENT_TOKEN = "<prohibited_content>"
