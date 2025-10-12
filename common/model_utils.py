from enum import Enum


class ModelFamily(Enum):
    # proprietary
    GEMINI = "gemini"
    OPENAI = "openai"
    # open source
    LLAMA = "llama"


def get_model_family(model_name: str) -> ModelFamily:
    if any([exp in model_name.lower() for exp in ["gpt", "o1", "o3", "o4"]]):
        return ModelFamily.OPENAI
    elif "gemini" in model_name.lower():
        return ModelFamily.GEMINI
    elif "llama" in model_name.lower():
        return ModelFamily.LLAMA
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
