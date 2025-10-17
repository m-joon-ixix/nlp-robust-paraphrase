import os
import argparse

from common.const import MODEL_CACHE_DIR
from common.json_utils import load_from_json, dump_to_json
from generate.open_src import get_model, get_tokenizer, get_generate_kwargs


def main(args):
    model_name = args.model_name
    print(f"Loading Model & Tokenizer '{model_name}' to CACHE DIR '{MODEL_CACHE_DIR}'")

    model = get_model(model_name)
    print(f"Model Loaded: {type(model)}")

    tokenizer = get_tokenizer(model_name)
    print(f"Tokenizer Loaded: {type(tokenizer)}")

    # remove default args from this model's generation_config.json
    generate_kwargs_keys = list(get_generate_kwargs(tokenizer, 1024, 0.5).keys())

    model_cache_dir = (
        f"{MODEL_CACHE_DIR}/models--{model_name.replace('/', '--')}/snapshots"
    )
    snapshot = os.listdir(model_cache_dir)[0]
    generation_config_path = f"{model_cache_dir}/{snapshot}/generation_config.json"
    generation_config = load_from_json(generation_config_path)

    filtered_generation_config = {}
    removed_args = []
    for k, v in generation_config.items():
        if "_token_id" not in k and k in generate_kwargs_keys:
            removed_args.append(k)
        else:
            filtered_generation_config[k] = v

    print(f"Following args removed from generation_config: {removed_args}")
    dump_to_json(generation_config_path, filtered_generation_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str)

    args = parser.parse_args()
    main(args)
