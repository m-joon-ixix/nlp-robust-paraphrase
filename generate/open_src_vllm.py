import os
import torch
from vllm import LLM, SamplingParams

from common.json_utils import load_from_json
from common.const import MODEL_CACHE_DIR
from common.model_utils import ModelFamily
from common.random_utils import get_seed
from common.secret_utils import load_secret
from generate.form_query import form_multichoice_queries


def get_vllm_model(name):
    # if finetuned:
    #     # VLLM & finetuned model
    #     print(f"Loading merged finetuned model with adapter from {adapter_path}")
    #     return LLdM(
    #         model=adapter_path,
    #         ...
    #     )
    # else:
    # VLLM & pure model
    print(f"Loading vLLM model {name}")
    os.environ["HF_TOKEN"] = load_secret("hf_key")
    model = LLM(
        model=name,
        trust_remote_code=True,
        download_dir=MODEL_CACHE_DIR,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        seed=get_seed(),
        gpu_memory_utilization=0.8,
        # enforce_eager=True,  # prevent repeated answers: https://github.com/vllm-project/vllm/issues/9448
    )

    return model


if __name__ == "__main__":
    data_list = load_from_json("./output/dataset/mmlu/paraphrased.json")[:5]
    prompts = form_multichoice_queries(data_list, "original", ModelFamily.LLAMA, 1)

    # prompts = [
    #     "Who is the Dodgers two-way player that both plays as a pitcher and a batter? You must only answer the name of this player.",
    #     "Who is the current president of the United States? You must only answer the name of him.",
    #     "Where is the capital of France? You must only answer the name of the city.",
    #     "What do you think about the future of AI? Briefly explain your thoughts.",
    # ]

    model = get_vllm_model("meta-llama/Llama-3.1-8B-Instruct")

    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=0.95,
        seed=get_seed(),
        max_tokens=1024,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    outputs = model.generate(prompts, sampling_params)

    for output in outputs:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"## Answer\n{generated_text}\n")
        print("-" * 100)
