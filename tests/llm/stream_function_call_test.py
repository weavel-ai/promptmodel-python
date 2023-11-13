import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from .constants import function_shemas
from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.types import LLMResponse, LLMStreamResponse
from promptmodel.utils.enums import ParsingType

html_output_format = """\
Output format:
<response type=str>
(value here)
</response>
"""

double_bracket_output_format = """\
Output format:
[[response type=str]]
(value here)
[[/response]]
"""

colon_output_format = """\
Output format:
response: (value here)

"""


def test_stream_with_functions(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: Generator[LLMStreamResponse, None, None] = llm.stream(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert final_response.function_call is not None, "function_call is None"

    assert (
        api_responses[0].choices[0]["message"]["content"] is None
    ), "content is not None"
    assert api_responses[0]["response_ms"] is not None, "response_ms is None"
    assert api_responses[0]["usage"] is not None, "usage is None"
    assert api_responses[0]["usage"]["prompt_tokens"] == 74, "prompt_tokens is not 74"

    # test logging
    llm_proxy = LLMProxy("test")

    mock_execute = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_execute.return_value = mock_response
    mocker.patch("promptmodel.llms.llm_proxy.AsyncAPIClient.execute", new=mock_execute)

    llm_proxy._sync_log_to_cloud(
        version_uuid="test",
        inputs={},
        api_response=api_responses[0],
        parsed_outputs={},
        metadata={},
    )

    mock_execute.assert_called_once()
    _, kwargs = mock_execute.call_args

    assert (
        kwargs["json"]["api_response"] == api_responses[0].to_dict_recursive()
    ), "api_response is not equal"


@pytest.mark.asyncio
async def test_astream_with_functions(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: Generator[LLMStreamResponse, None, None] = llm.astream(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    async for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert final_response.function_call is not None, "function_call is None"

    assert (
        api_responses[0].choices[0]["message"]["content"] is None
    ), "content is not None"
    assert api_responses[0]["response_ms"] is not None, "response_ms is None"
    assert api_responses[0]["usage"] is not None, "usage is None"
    assert api_responses[0]["usage"]["prompt_tokens"] == 74, "prompt_tokens is not 74"

    # test logging
    llm_proxy = LLMProxy("test")

    mock_execute = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_execute.return_value = mock_response
    mocker.patch("promptmodel.llms.llm_proxy.AsyncAPIClient.execute", new=mock_execute)

    await llm_proxy._async_log_to_cloud(
        version_uuid="test",
        inputs={},
        api_response=api_responses[0],
        parsed_outputs={},
        metadata={},
    )

    mock_execute.assert_called_once()
    _, kwargs = mock_execute.call_args

    assert (
        kwargs["json"]["api_response"] == api_responses[0].to_dict_recursive()
    ), "api_response is not equal"


def test_stream_and_parse_with_functions(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: Generator[LLMStreamResponse, None, None] = llm.stream_and_parse(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
        parsing_type=None,
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert final_response.function_call is not None, "function_call is None"

    assert (
        api_responses[0].choices[0]["message"]["content"] is None
    ), "content is not None"
    assert final_response.function_call is not None, "function_call is None"

    # Not call function, parsing case
    messages = [
        {
            "role": "user",
            "content": "Hello, How are you?\n" + html_output_format,
        }
    ]
    stream_res: Generator[LLMStreamResponse, None, None] = llm.stream_and_parse(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert final_response.function_call is None, "function_call is not None"
    assert final_response.parsed_outputs != {}, "parsed_outputs is empty dict"

    assert (
        api_responses[0].choices[0]["message"]["content"] is None
    ), "content is not None"
    assert final_response.function_call is not None, "function_call is None"


@pytest.mark.asyncio
async def test_astream_and_parse_with_functions(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: Generator[LLMStreamResponse, None, None] = llm.astream_and_parse(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
        parsing_type=None,
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    async for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"

    assert (
        api_responses[0].choices[0]["message"]["content"] is None
    ), "content is not None"
    assert final_response.function_call is not None, "function_call is None"

    # Not call function, parsing case
    messages = [
        {
            "role": "user",
            "content": "Hello, How are you?\n" + html_output_format,
        }
    ]
    stream_res: Generator[LLMStreamResponse, None, None] = llm.astream_and_parse(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    async for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert final_response.function_call is None, "function_call is not None"
    assert final_response.parsed_outputs != {}, "parsed_outputs is empty dict"

    assert (
        api_responses[0].choices[0]["message"]["content"] is None
    ), "content is not None"
    assert final_response.function_call is not None, "function_call is None"
