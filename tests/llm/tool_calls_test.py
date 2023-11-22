import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from ..constants import function_shemas
from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import *
from promptmodel.types.enums import ParsingType

html_output_format = """\
You must follow the provided output format. Keep the string between <> as it is.
Output format:
<response type=str>
(value here)
</response>
"""

double_bracket_output_format = """\
You must follow the provided output format. Keep the string between [[ ]] as it is.
Output format:
[[response type=str]]
(value here)
[[/response]]
"""

colon_output_format = """\
You must follow the provided output format. Keep the string before : as it is
Output format:
response type=str: (value here)

"""

tool_schemas = [{"type": "function", "function": schema} for schema in function_shemas]


def test_run_with_tools(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    res: LLMResponse = llm.run(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
    )

    assert res.error is None, "error is not None"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "tool_calls"
    ), "finish_reason is not tool_calls"
    print(res.api_response.model_dump())
    print(res.__dict__)
    assert res.tool_calls is not None, "tool_calls is None"
    assert isinstance(res.tool_calls[0], ChatCompletionMessageToolCall)

    messages = [{"role": "user", "content": "Hello, How are you?"}]

    res: LLMResponse = llm.run(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
    )

    assert res.error is None, "error is not None"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "stop"
    ), "finish_reason is not stop"

    assert res.tool_calls is None, "tool_calls is not None"
    assert res.raw_output is not None, "raw_output is None"


@pytest.mark.asyncio
async def test_arun_with_tools(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    res: LLMResponse = await llm.arun(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
    )

    assert res.error is None, "error is not None"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "tool_calls"
    ), "finish_reason is not tool_calls"

    assert res.tool_calls is not None, "tool_calls is None"
    assert isinstance(res.tool_calls[0], ChatCompletionMessageToolCall)

    messages = [{"role": "user", "content": "Hello, How are you?"}]

    res: LLMResponse = await llm.arun(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
    )

    assert res.error is None, "error is not None"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "stop"
    ), "finish_reason is not stop"

    assert res.tool_calls is None, "tool_calls is not None"
    assert res.raw_output is not None, "raw_output is None"


def test_run_and_parse_with_tools(mocker):
    # With parsing_type = None
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    res: LLMResponse = llm.run_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=None,
    )

    assert res.error is False, "error is not False"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "tool_calls"
    ), "finish_reason is not tool_calls"

    assert res.tool_calls is not None, "tool_calls is None"
    assert isinstance(res.tool_calls[0], ChatCompletionMessageToolCall)

    messages = [{"role": "user", "content": "Hello, How are you?"}]

    res: LLMResponse = llm.run_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=None,
    )

    assert res.error is False, "error is not False"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "stop"
    ), "finish_reason is not stop"

    assert res.tool_calls is None, "tool_calls is not None"
    assert res.raw_output is not None, "raw_output is None"

    # with parsing_type = "HTML"

    messages = [
        {
            "role": "user",
            "content": "What is the weather like in Boston? \n" + html_output_format,
        }
    ]
    llm = LLM()
    res: LLMResponse = llm.run_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    # 1. Output 지키고 function call ->  (Pass)
    # 2. Output 지키고 stop -> OK
    # 3. Output 무시하고 function call -> OK (function call이 나타나면 파싱을 하지 않도록 수정)

    # In this case, error is True because the output is not in the correct format
    assert res.error is False, "error is not False"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "tool_calls"
    ), "finish_reason is not tool_calls"

    assert res.tool_calls is not None, "tool_calls is None"
    assert isinstance(res.tool_calls[0], ChatCompletionMessageToolCall)

    assert res.parsed_outputs is None, "parsed_outputs is not empty"

    messages = [
        {
            "role": "user",
            "content": "Hello, How are you?\n" + html_output_format,
        }
    ]

    res: LLMResponse = llm.run_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    print(res.__dict__)

    if not "str" in res.raw_output:
        # if "str" in res.raw_output, it means that LLM make mistakes
        assert res.error is False, "error is not False"
        assert res.parsed_outputs is not None, "parsed_outputs is None"

    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "stop"
    ), "finish_reason is not stop"

    assert res.tool_calls is None, "tool_calls is not None"
    assert res.raw_output is not None, "raw_output is None"


@pytest.mark.asyncio
async def test_arun_and_parse_with_tools(mocker):
    # With parsing_type = None
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    res: LLMResponse = await llm.arun_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=None,
    )

    assert res.error is False, "error is not False"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "tool_calls"
    ), "finish_reason is not tool_calls"
    print(res)
    assert res.tool_calls is not None, "tool_calls is None"
    assert isinstance(res.tool_calls[0], ChatCompletionMessageToolCall)

    messages = [{"role": "user", "content": "Hello, How are you?"}]

    res: LLMResponse = await llm.arun_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=None,
    )

    assert res.error is False, "error is not False"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "stop"
    ), "finish_reason is not stop"

    assert res.tool_calls is None, "tool_calls is not None"
    assert res.raw_output is not None, "raw_output is None"

    # with parsing_type = "HTML"

    messages = [
        {
            "role": "user",
            "content": "What is the weather like in Boston? \n" + html_output_format,
        }
    ]
    llm = LLM()
    res: LLMResponse = await llm.arun_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    # In this case, error is False becuase if tool_calls, parsing is not performed
    assert res.error is False, "error is not False"
    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "tool_calls"
    ), "finish_reason is not tool_calls"

    assert res.tool_calls is not None, "tool_calls is None"
    assert isinstance(res.tool_calls[0], ChatCompletionMessageToolCall)

    assert res.parsed_outputs is None, "parsed_outputs is not empty"

    messages = [
        {
            "role": "user",
            "content": "Hello, How are you?\n" + html_output_format,
        }
    ]

    res: LLMResponse = await llm.arun_and_parse(
        messages=messages,
        tools=tool_schemas,
        model="gpt-4-1106-preview",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    if not "str" in res.raw_output:
        assert res.error is False, "error is not False"
        assert res.parsed_outputs is not None, "parsed_outputs is None"

    assert res.api_response is not None, "api_response is None"
    assert (
        res.api_response.choices[0].finish_reason == "stop"
    ), "finish_reason is not stop"

    assert res.tool_calls is None, "tool_calls is not None"
    assert res.raw_output is not None, "raw_output is None"
