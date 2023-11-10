import os
import sys
import yaml
import asyncio
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Tuple, List, Union, Optional, Coroutine
from litellm import token_counter

from promptmodel.apis.base import AsyncAPIClient
from promptmodel.database.crud import (
    get_latest_version_prompts,
    get_deployed_prompts,
)
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils import logger
from promptmodel.utils.random_utils import select_version
from promptmodel.promptmodel_init import CacheManager, update_deployed_db


async def fetch_prompts(name) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """fetch prompts.

    Args:
        name (str): name of promtpmodel

    Returns:
        Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]: (prompts, version_detail)
    """
    # Check dev_branch activate
    config = read_config()
    if "dev_branch" in config and config["dev_branch"]["initializing"] == True:
        return [], {}
    elif "dev_branch" in config and config["dev_branch"]["online"] == True:
        # get prompt from local DB
        prompt_rows, version_detail = get_latest_version_prompts(name)
        if prompt_rows is None:
            return [], {}
        return [
            {"role": prompt.role, "content": prompt.content} for prompt in prompt_rows
        ], version_detail
    else:
        if (
            "project" in config
            and "use_cache" in config["project"]
            and config["project"]["use_cache"] == True
        ):
            cache_manager = CacheManager()
            # call update_local API in background task
            cache_update_thread = Thread(
                target=cache_manager.cache_update_background_task, args=(config,)
            )
            cache_update_thread.daemon = True
            cache_update_thread.start()

            # get prompt from local DB by ratio
            prompt_rows, version_detail = get_deployed_prompts(name)
            if prompt_rows is None:
                return [], {}

            return [
                {"role": prompt.role, "content": prompt.content}
                for prompt in prompt_rows
            ], version_detail

        else:
            try:
                prompts_data = await AsyncAPIClient.execute(
                    method="GET",
                    path="/fetch_published_prompt_model_version",
                    params={"prompt_model_name": name},
                    use_cli_key=False,
                )
                prompts_data = prompts_data.json()
            except Exception as e:
                raise e
            prompt_model_versions = prompts_data["prompt_model_versions"]
            prompts = prompts_data["prompts"]
            for version in prompt_model_versions:
                if version["is_published"] is True:
                    version["ratio"] = 1.0
            selected_version = select_version(prompt_model_versions)

            prompt_rows = list(
                filter(
                    lambda prompt: str(prompt["version_uuid"])
                    == str(selected_version["uuid"]),
                    prompts,
                )
            )
            # sort prompt_rows by step
            prompt_rows = sorted(prompt_rows, key=lambda prompt: prompt["step"])

            version_detail = {
                "model": selected_version["model"],
                "uuid": selected_version["uuid"],
                "parsing_type": selected_version["parsing_type"],
                "output_keys": selected_version["output_keys"],
            }

            if prompt_rows is None:
                return [], {}

            return [
                {"role": prompt["role"], "content": prompt["content"]}
                for prompt in prompt_rows
            ], version_detail


def set_inputs_to_prompts(inputs: Dict[str, Any], prompts: List[Dict[str, str]]):
    messages = [
        {"content": prompt["content"].format(**inputs), "role": prompt["role"]}
        for prompt in prompts
    ]
    return messages


def num_tokens_for_messages(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613"
) -> int:
    tokens_per_message = 0
    tokens_per_name = 0
    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1

    if model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1

    if model.endswith("-0613") or model == "gpt-3.5-turbo-16k":
        tokens_per_message = 3
        tokens_per_name = 1
    sum = 0
    sum = token_counter(model=model, messages=messages)
    for message in messages:
        sum += tokens_per_message
        if "name" in message:
            sum += tokens_per_name
    return sum


def num_tokens_from_functions_input(
    functions: List[Any], model="gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of functions."""

    num_tokens = 0
    for function in functions:
        function_tokens = token_counter(model=model, text=function["name"])
        function_tokens += token_counter(model=model, text=function["description"])

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for properties_key in parameters["properties"]:
                    function_tokens += token_counter(model=model, text=properties_key)
                    v = parameters["properties"][properties_key]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += token_counter(
                                model=model, text=v["type"]
                            )
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += token_counter(
                                model=model, text=v["description"]
                            )
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += token_counter(model=model, text=o)
                        else:
                            print(f"Warning: not supported field {field}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens


def num_tokens_from_function_call_output(
    function_call_output: Dict[str, str] = {}, model="gpt-3.5-turbo-0613"
) -> int:
    num_tokens = 1
    num_tokens += token_counter(model=model, text=function_call_output["name"])
    if "arguments" in function_call_output:
        num_tokens += token_counter(model=model, text=function_call_output["arguments"])
    return num_tokens


import asyncio


def run_async_in_sync(coro: Coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        # loop.close()
        return result

    if loop.is_running():
        # nest_asyncio.apply already done
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)
