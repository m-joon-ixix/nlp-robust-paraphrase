import random
import time
from typing import List
from tqdm import tqdm
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai.errors import APIError
from snowflake.snowpark import Session
from snowflake.cortex import complete

from common.const import FAILED_TOKEN, PROHIBITED_CONTENT_TOKEN
from common.model_utils import ModelFamily, get_model_family
from common.random_utils import get_seed
from common.secret_utils import load_secret
from common.slack_utils import slack_notify

RETRY_INTERVAL = 3

SNOWFLAKE_CONFIG_KEYS = [
    "account",
    "user",
    "password",
    "role",
    "database",
    "schema",
    "warehouse",
]


def batch_query_api(
    query_list: list,
    model_name: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> List[str]:
    print_prompt_example(query_list[0], model_name)

    kwargs = {}
    if get_model_family(model_name) == ModelFamily.SNOWFLAKE:
        kwargs["snowflake_session"] = _get_snowflake_session()

    responses = [
        query_api(query, model_name, max_new_tokens, temperature, **kwargs)
        for query in tqdm(query_list, desc=f"Querying {model_name} API")
    ]

    if "snowflake_session" in kwargs:
        kwargs["snowflake_session"].close()

    return responses


def query_api(
    query, model_name: str, max_tokens: int, temperature: float, **kwargs
) -> str:
    model_family = get_model_family(model_name)
    if model_family == ModelFamily.GEMINI:
        return _query_gemini(query, model_name, max_tokens, temperature)
    elif model_family == ModelFamily.SNOWFLAKE:
        return _query_snowflake(
            query, model_name, max_tokens, temperature, kwargs["snowflake_session"]
        )
    else:
        raise NotImplementedError()


def _query_gemini(query, model_name: str, max_tokens: int, temperature: float) -> str:
    api_keys = get_api_keys(get_model_family(model_name))
    # rotate multiple API keys to avoid the rate limit error
    clients = [genai.Client(api_key=api_key) for api_key in api_keys]
    client_idx = random.randint(0, len(clients) - 1)

    retry_count = 10
    for _ in range(retry_count):
        try:
            client = clients[client_idx]
            config = GenerateContentConfig(
                temperature=temperature,
                top_p=1.0,
                # candidate_count=1,
                max_output_tokens=max_tokens,
                seed=get_seed(),
            )
            response = client.models.generate_content(
                contents=query,
                model=model_name,
                config=config,
            )
            msg = response.text
            if msg is None or len(msg.strip()) == 0:
                finish_reason = _get_gemini_finish_reason(response)
                if finish_reason == "MAX_TOKENS":
                    max_tokens *= 2
                if finish_reason == "PROHIBITED_CONTENT":
                    return PROHIBITED_CONTENT_TOKEN

                print(f"Message empty due to: {finish_reason} => retrying...")
                continue

            time.sleep(3)  # sleep to not exceed rate limit
            return msg
        except Exception as e:
            if isinstance(e, APIError) and e.code == 429:
                print(f"[Client {client_idx}]", end=" ")
                _handle_gemini_429_error(e)
            else:
                print(f"[Client {client_idx}] Retrying due to Error: ", e)
                time.sleep(RETRY_INTERVAL)
        finally:
            client_idx = (client_idx + 1) % len(clients)

    # if reached to this stage, it means every trial has failed
    return FAILED_TOKEN


def _query_snowflake(
    query, model_name: str, max_tokens: int, temperature: float, session: Session
) -> str:
    response = complete(
        model=model_name,
        prompt=query,
        options={"max_tokens": max_tokens, "temperature": temperature},
        session=session,
    )

    time.sleep(1)  # sleep to not impose high throughput
    return response


def print_prompt_example(prompt, model_name: str):
    print(f"An example of prompt:")
    print("-" * 100)

    model_family = get_model_family(model_name)
    # check the `if`, `elif` statements in the method `form_query()` for each case
    if model_family == ModelFamily.OPENAI:
        print(prompt[0]["content"][0]["text"])
    elif model_family == ModelFamily.GEMINI:
        print(prompt[0]["parts"][0]["text"])
    elif model_family == ModelFamily.SNOWFLAKE:
        print(f"# System\n{prompt[0]['content']}\n\n# User\n{prompt[1]['content']}")
    else:
        print(
            f"Warning from print_prompt_example() - Unexpected model name: {model_name}"
        )

    print("-" * 100)


def _get_gemini_finish_reason(response) -> str:
    try:
        if response.candidates:
            return response.candidates[0].finish_reason.value
        elif response.prompt_feedback:
            return response.prompt_feedback.block_reason.value
        else:
            return str(response)
    except Exception:
        return str(response)


def _handle_gemini_429_error(e: Exception):
    # HTTP 429: Too Many Requests => Exceeded Quota
    try:
        debug_info = {
            k: v
            for k, v in e.details["error"]["details"][0]["violations"][0].items()
            if k in ["quotaId", "quotaValue"]
        }
        retry_delay = e.details["error"]["details"][-1]["retryDelay"]
        debug_info["retryDelay"] = retry_delay
        print(f"Retrying due to Gemini 429 Error: {debug_info}")
    except Exception:  # if error occurred while parsing
        print("Retrying due to Gemini 429 Error. Exception occurred during parsing:", e)

    # time.sleep(int(retry_delay.replace("s", "")) + 1)
    # NOTE: just sleep for a fixed time since multiple API keys are being rotated
    time.sleep(RETRY_INTERVAL)


def get_api_keys(model_family: ModelFamily) -> List[str]:
    api_keys = load_secret("api_keys", print_log=False)[model_family.value]
    if len(api_keys) == 0:
        raise AssertionError(f"No API keys found for {model_family.value}")

    return api_keys


def _get_snowflake_session() -> Session:
    snowflake_config = load_secret("snowflake", print_log=False)
    connection_params = {}
    for key in SNOWFLAKE_CONFIG_KEYS:
        connection_params[key] = snowflake_config.get(key, "")

    return Session.builder.configs(connection_params).create()


def check_generate_failure(
    responses: List[str], model_name: str, output_path: str, required_str=None
):
    failed_indices = []
    prohibited_content_indices = []
    required_str_not_found_indices = []

    for idx, response in enumerate(responses):
        if response == FAILED_TOKEN:
            failed_indices.append(idx)
        elif response == PROHIBITED_CONTENT_TOKEN:
            prohibited_content_indices.append(idx)
        elif required_str is not None and required_str not in response:
            required_str_not_found_indices.append(idx)

    notify_params = {"output_path": output_path, "model_name": model_name}

    if len(failed_indices) > 0:
        slack_notify(
            f"Failed to generate for examples at these indices: {failed_indices}",
            **notify_params,
        )

    if len(prohibited_content_indices) > 0:
        slack_notify(
            f"Generate request was blocked due to PROHIBITED_CONTENT for examples at these indices: {prohibited_content_indices}",
            **notify_params,
        )

    if len(required_str_not_found_indices) > 0:
        slack_notify(
            f"Required string '{required_str}' not found for examples at these indices: {required_str_not_found_indices}",
            **notify_params,
        )
