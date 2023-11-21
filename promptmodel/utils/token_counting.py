from typing import Any, Dict, List
from litellm import token_counter


def set_inputs_to_prompts(inputs: Dict[str, Any], prompts: List[Dict[str, str]]):
    messages = [
        {"content": prompt["content"].format(**inputs), "role": prompt["role"]}
        for prompt in prompts
    ]
    return messages


def num_tokens_for_messages(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613"
) -> int:
    # tokens_per_message = 0
    # tokens_per_name = 0
    # if model.startswith("gpt-3.5-turbo"):
    #     tokens_per_message = 4
    #     tokens_per_name = -1

    # if model.startswith("gpt-4"):
    #     tokens_per_message = 3
    #     tokens_per_name = 1

    # if model.endswith("-0613") or model == "gpt-3.5-turbo-16k":
    #     tokens_per_message = 3
    #     tokens_per_name = 1
    # sum = 0
    processed_messages = [
        {**message, "function_call": str(message["function_call"])}
        if "function_call" in message
        else message
        for message in messages
    ]
    sum = token_counter(model=model, messages=processed_messages)
    # for message in messages:
    #     sum += tokens_per_message
    #     if "name" in message:
    #         sum += tokens_per_name
    return sum


def num_tokens_for_messages_for_each(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613"
) -> List[int]:
    processed_messages = [
        {**message, "function_call": str(message["function_call"])}
        if "function_call" in message
        else message
        for message in messages
    ]
    return [
        token_counter(model=model, messages=[message]) for message in processed_messages
    ]


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
