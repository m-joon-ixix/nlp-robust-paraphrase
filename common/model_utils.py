from enum import Enum
from typing import List


class ModelFamily(Enum):
    # proprietary
    GEMINI = "gemini"
    SNOWFLAKE = "snowflake"
    # open source
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"

    def is_open_src(self) -> bool:
        return self in [ModelFamily.LLAMA, ModelFamily.QWEN, ModelFamily.MISTRAL]


OPEN_SRC_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

PROPRIETARY_MODELS = [
    "gemini-2.5-flash-lite",
    "claude-3-5-sonnet",
    "llama3.1-405b",  # accessed via Snowflake API
    "deepseek-r1",
]

PARAPHRASE_MODEL_MAP = {
    "gemini": "gemini-2.5-flash-lite",
    "claude": "claude-3-5-sonnet",
}

# RPD: Requests Per Day
MODEL_TO_RPD_LIMIT = {
    "gemini-2.5-flash-lite": 1000,
    "claude-3-5-sonnet": 200000,
}


def get_model_family(model_name: str) -> ModelFamily:
    if any([exp in model_name.lower() for exp in ["openai", "claude"]]):
        return ModelFamily.SNOWFLAKE
    elif "gemini" in model_name.lower():
        return ModelFamily.GEMINI
    elif model_name in PROPRIETARY_MODELS:
        return ModelFamily.SNOWFLAKE
    elif "llama" in model_name.lower():
        return ModelFamily.LLAMA
    elif "qwen" in model_name.lower():
        return ModelFamily.QWEN
    elif "mistral" in model_name.lower():
        return ModelFamily.MISTRAL
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def model_name_to_dirname(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


# To use as sorting key in: `sorted([...], key=model_sort_key)`
def model_sort_key(model_name: str):
    model_family = model_name.split("-")[0]
    param_size = int(
        [s for s in model_name.split("-") if s.endswith("B")][0].replace("B", "")
    )

    return (model_family, param_size)


def models_to_finetune() -> List[str]:
    # if model param size is smaller than 10B
    return [
        model_name
        for model_name in OPEN_SRC_MODELS
        if model_sort_key(model_name)[1] < 10
    ]
