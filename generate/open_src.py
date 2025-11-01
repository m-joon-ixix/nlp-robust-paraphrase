import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Optional

from common.const import MODEL_CACHE_DIR
from common.secret_utils import load_secret


def batch_query_open_src(
    query_list: list,
    model_name: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    batch_size: int = 8,
    peft_dir: Optional[str] = None,
) -> List[str]:
    model = get_model(model_name, peft_dir=peft_dir)
    tokenizer = get_tokenizer(model_name)
    model.eval()

    generate_kwargs = get_generate_kwargs(tokenizer, max_new_tokens, temperature)
    print(f"Generate KwArgs: {generate_kwargs}")

    print(f"An example of prompt:")
    print("-" * 100)
    # check the `else` statement in the method `form_query()`
    print(query_list[0][-1]["content"])
    print("-" * 100)

    if peft_dir:
        print(f"Using adapters from peft_dir: {peft_dir}")

    responses = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(query_list), batch_size),
            desc=f"Batch Generation (model: {model_name}, batch size: {batch_size})",
        ):
            inputs = _get_model_inputs(
                query_list[i : (i + batch_size)], model, tokenizer
            )

            outputs = model.generate(**inputs, **generate_kwargs)
            sequences = outputs.sequences[:, inputs["input_ids"].shape[-1] :].cpu()
            _responses = tokenizer.batch_decode(sequences, skip_special_tokens=True)

            responses.extend(_responses)

            # clear memory after each batch to prevent memory usage from keep increasing
            del outputs, inputs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return responses


def get_model(model_name: str, peft_dir: Optional[str] = None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        **_hf_loading_kwargs(),
    )

    if peft_dir is None:
        return model
    else:
        return PeftModel.from_pretrained(model, peft_dir)


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, **_hf_loading_kwargs())

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _hf_loading_kwargs() -> dict:
    return {
        "token": load_secret("hf_key"),
        "cache_dir": MODEL_CACHE_DIR,
        "trust_remote_code": True,
    }


def _get_model_inputs(query_list: list, model, tokenizer):
    text_list = tokenizer.apply_chat_template(
        query_list,
        add_generation_prompt=True,
        tokenize=False,
    )

    return tokenizer(text_list, return_tensors="pt", padding=True).to(model.device)


def get_generate_kwargs(tokenizer, max_new_tokens: int, temperature: float) -> dict:
    kwargs = {
        "do_sample": True if temperature != 0.0 else False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }

    # To prevent warning messages, set temperature only in sample-based generation
    if kwargs["do_sample"]:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95

    return kwargs
