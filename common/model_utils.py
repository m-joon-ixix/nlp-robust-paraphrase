from enum import Enum


class ModelFamily(Enum):
    # proprietary
    GEMINI = "gemini"
    SNOWFLAKE = "snowflake"
    OPENAI = "openai"
    # open source
    LLAMA = "llama"


OPEN_SRC_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Small-24B-Instruct-2501",
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
    elif "llama" in model_name.lower():
        return ModelFamily.LLAMA
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
