import os
import sys
import yaml
import asyncio
from threading import Thread
from concurrent.futures import Future
from typing import Any, Dict, Tuple, List, Union, Optional, Coroutine
from litellm import token_counter

from promptmodel.database.crud import (
    get_latest_version_prompts,
    get_deployed_prompts,
    update_deployed_cache,
)
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils import logger
from promptmodel.apis.base import APIClient, AsyncAPIClient


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
            {"role": prompt.role, "content": prompt.content}
            for prompt in prompt_rows
        ], version_detail
    else:
        if (
            "project" in config
            and "use_cache" in config["project"]
            and config["project"]["use_cache"] == True
        ):
            # call update_local API in background task
            asyncio.create_task(update_deployed_db(config))
            # get prompt from local DB by ratio
            prompt_rows, version_detail = get_deployed_prompts(name)
            if prompt_rows is None:
                return [], {}
        else:
            await update_deployed_db(config) # wait for update local DB cache
            prompt_rows, version_detail = get_deployed_prompts(name)
            if prompt_rows is None:
                return [], {}

        return [
            {"role": prompt.role, "content": prompt.content} for prompt in prompt_rows
        ], version_detail


async def update_deployed_db(config):
    if "project" not in config or "version" not in config["project"]:
        cached_project_version = "0.0.0"
    else:
        cached_project_version = config["project"]["version"]
    try:
        res = await AsyncAPIClient.execute(
            method="GET",
            path="/check_update",
            params={"cached_version": cached_project_version},
            use_cli_key=False,
        )
        res = res.json()
        if res["need_update"]:
            # update local DB with res['project_status']
            project_status = res["project_status"]
            await update_deployed_cache(project_status)
            upsert_config({"version": res["version"]}, section="project")
        else:
            upsert_config({"version": res["version"]}, section="project")
    except Exception as exception:
        logger.error(f"Deployment cache update error: {exception}")


def set_inputs_to_prompts(inputs: Dict[str, Any], prompts: List[Dict[str, str]]):
    messages = [
        {"content": prompt["content"].format(**inputs), "role": prompt["role"]}
        for prompt in prompts
    ]
    return messages

def num_tokens_for_messages(messages: List[Dict[str, str]], model : str ="gpt-3.5-turbo-0613") -> int:
    tokens_per_message = 0
    tokens_per_name = 0
    if model.startswith("gpt-3.5-turbo"): 
        tokens_per_message = 4
        tokens_per_name = -1
    
    if (model.startswith("gpt-4")):
        tokens_per_message = 3
        tokens_per_name = 1
    
    if (model.endswith("-0613") or model == "gpt-3.5-turbo-16k"): 
        tokens_per_message = 3
        tokens_per_name = 1
    sum = 0
    sum = token_counter(model=model, messages=messages)
    for message in messages:
        sum += tokens_per_message
        if "name" in message:
            sum += tokens_per_name
    return sum

def num_tokens_from_functions_input(functions: List[Any], model="gpt-3.5-turbo-0613") -> int:
        """Return the number of tokens used by a list of functions."""
        
        num_tokens = 0
        for function in functions:
            function_tokens = token_counter(model=model, text=function['name'])
            function_tokens += token_counter(model=model, text=function['description'])
            
            if 'parameters' in function:
                parameters = function['parameters']
                if 'properties' in parameters:
                    for properties_key in parameters['properties']:
                        function_tokens += token_counter(model=model, text=properties_key)
                        v = parameters['properties'][properties_key]
                        for field in v:
                            if field == 'type':
                                function_tokens += 2
                                function_tokens += token_counter(model=model, text=v['type'])
                            elif field == 'description':
                                function_tokens += 2
                                function_tokens += token_counter(model=model, text=v['description'])
                            elif field == 'enum':
                                function_tokens -= 3
                                for o in v['enum']:
                                    function_tokens += 3
                                    function_tokens += token_counter(model=model, text=o)
                            else:
                                print(f"Warning: not supported field {field}")
                    function_tokens += 11

            num_tokens += function_tokens

        num_tokens += 12 
        return num_tokens
    
def num_tokens_from_function_call_output(
    function_call_output: Dict[str, str] = {},
    model="gpt-3.5-turbo-0613"
) -> int:
    num_tokens = 1
    num_tokens += token_counter(model=model, text=function_call_output['name'])
    if 'arguments' in function_call_output:
        num_tokens += token_counter(model=model, text=function_call_output['arguments'])
    return num_tokens

import asyncio

def run_async_in_sync(coro : Coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    
    if loop.is_running():
        future = Future()
        def target(fut : Future, coro : Coroutine):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coro)
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)
            finally:
                new_loop.close()
                
        thread = Thread(target=target, args=(future, coro), daemon=True)
        thread.start()
        thread.join()
        return future.result()
    else:
        return loop.run_until_complete(coro)