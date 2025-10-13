from enum import Enum


class ModelFamily(Enum):
    # proprietary
    GEMINI = "gemini"
    SNOWFLAKE = "snowflake"
    OPENAI = "openai"
    # open source
    LLAMA = "llama"


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
