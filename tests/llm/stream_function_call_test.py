import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import nest_asyncio
from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from ..constants import function_shemas
from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import *
from promptmodel.types.enums import ParsingType
from promptmodel.utils.async_utils import run_async_in_sync

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
    chunks: List[LLMStreamResponse] = []
    for res in stream_res:
        chunks.append(res)
        if res.error:
            error_count += 1
        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert (
        getattr(final_response.api_response.choices[0].message, "function_call", None)
        is not None
    ), "function_call is None"
    assert isinstance(
        final_response.api_response.choices[0].message.function_call, FunctionCall
    )

    assert len([c for c in chunks if c.function_call is not None]) > 0
    assert isinstance(chunks[1].function_call, ChoiceDeltaFunctionCall) is True

    assert api_responses[0].choices[0].message.content is None, "content is not None"
    assert api_responses[0]._response_ms is not None, "response_ms is None"
    assert api_responses[0].usage is not None, "usage is None"

    # test logging
    llm_proxy = LLMProxy("test")

    mock_execute = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_execute.return_value = mock_response
    mocker.patch("promptmodel.llms.llm_proxy.AsyncAPIClient.execute", new=mock_execute)

    nest_asyncio.apply()
    run_async_in_sync(
        llm_proxy._async_log_to_cloud(
            log_uuid="test",
            version_uuid="test",
            inputs={},
            api_response=api_responses[0],
            parsed_outputs={},
            metadata={},
        )
    )

    mock_execute.assert_called_once()
    _, kwargs = mock_execute.call_args
    api_response_dict = api_responses[0].model_dump()
    api_response_dict.update({"response_ms": api_responses[0]._response_ms})

    assert (
        kwargs["json"]["api_response"] == api_response_dict
    ), "api_response is not equal"


@pytest.mark.asyncio
async def test_astream_with_functions(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: AsyncGenerator[LLMStreamResponse, None] = llm.astream(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    chunks: List[LLMStreamResponse] = []
    async for res in stream_res:
        chunks.append(res)
        if res.error:
            error_count += 1
        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert (
        getattr(final_response.api_response.choices[0].message, "function_call", None)
        is not None
    ), "function_call is None"
    assert isinstance(
        final_response.api_response.choices[0].message.function_call, FunctionCall
    )
    assert len([c for c in chunks if c.function_call is not None]) > 0
    assert isinstance(chunks[1].function_call, ChoiceDeltaFunctionCall)

    assert api_responses[0].choices[0].message.content is None, "content is not None"
    assert api_responses[0]._response_ms is not None, "response_ms is None"
    assert api_responses[0].usage is not None, "usage is None"
    # assert api_responses[0].usage.prompt_tokens == 74, "prompt_tokens is not 74"

    # test logging
    llm_proxy = LLMProxy("test")

    mock_execute = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_execute.return_value = mock_response
    mocker.patch("promptmodel.llms.llm_proxy.AsyncAPIClient.execute", new=mock_execute)
    nest_asyncio.apply()
    await llm_proxy._async_log_to_cloud(
        log_uuid="test",
        version_uuid="test",
        inputs={},
        api_response=api_responses[0],
        parsed_outputs={},
        metadata={},
    )

    mock_execute.assert_called_once()
    _, kwargs = mock_execute.call_args
    api_response_dict = api_responses[0].model_dump()
    api_response_dict.update({"response_ms": api_responses[0]._response_ms})

    assert (
        kwargs["json"]["api_response"] == api_response_dict
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
    chunks: List[LLMStreamResponse] = []
    for res in stream_res:
        chunks.append(res)
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"
    assert (
        getattr(final_response.api_response.choices[0].message, "function_call", None)
        is not None
    ), "function_call is None"
    assert isinstance(
        final_response.api_response.choices[0].message.function_call, FunctionCall
    )
    assert len([c for c in chunks if c.function_call is not None]) > 0
    assert isinstance(chunks[1].function_call, ChoiceDeltaFunctionCall)

    assert api_responses[0].choices[0].message.content is None, "content is not None"

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
        model="gpt-4-1106-preview",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    error_count = 0
    error_log = ""
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    chunks: List[LLMStreamResponse] = []
    for res in stream_res:
        chunks.append(res)
        if res.error:
            error_count += 1
            error_log = res.error_log
            print("ERROR")
        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
            final_response = res

    if not "str" in api_responses[0].choices[0].message.content:
        # if "str" in content, just LLM make mistake in generation.
        assert (
            error_count == 0
        ), f"error_count is not 0, {error_log}, {api_responses[0].model_dump()}"
        assert final_response.parsed_outputs is not None, "parsed_outputs is None"
    assert len(api_responses) == 1, "api_count is not 1"
    assert (
        getattr(final_response.api_response.choices[0].message, "function_call", None)
        is None
    ), "function_call is not None"
    assert len([c for c in chunks if c.function_call is not None]) == 0

    assert api_responses[0].choices[0].message.content is not None, "content is None"


@pytest.mark.asyncio
async def test_astream_and_parse_with_functions(mocker):
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: AsyncGenerator[LLMStreamResponse, None] = llm.astream_and_parse(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
        parsing_type=None,
    )

    error_count = 0
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    chunks: List[LLMStreamResponse] = []
    async for res in stream_res:
        chunks.append(res)
        if res.error:
            error_count += 1

        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
            final_response = res

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"

    assert api_responses[0].choices[0].message.content is None, "content is not None"
    assert (
        getattr(final_response.api_response.choices[0].message, "function_call", None)
        is not None
    ), "function_call is None"
    assert isinstance(
        final_response.api_response.choices[0].message.function_call, FunctionCall
    )
    assert len([c for c in chunks if c.function_call is not None]) > 0
    assert isinstance(chunks[1].function_call, ChoiceDeltaFunctionCall)

    # Not call function, parsing case
    messages = [
        {
            "role": "user",
            "content": "Hello, How are you?\n" + html_output_format,
        }
    ]
    stream_res: AsyncGenerator[LLMStreamResponse, None] = llm.astream_and_parse(
        messages=messages,
        functions=function_shemas,
        model="gpt-3.5-turbo-0613",
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )

    error_count = 0
    error_log = ""
    api_responses: List[ModelResponse] = []
    final_response: Optional[LLMStreamResponse] = None
    chunks: List[LLMStreamResponse] = []
    async for res in stream_res:
        chunks.append(res)
        if res.error:
            error_count += 1
            error_log = res.error_log

        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
            final_response = res

    if not "str" in api_responses[0].choices[0].message.content:
        # if "str" in content, just LLM make mistake in generation.
        assert (
            error_count == 0
        ), f"error_count is not 0, {error_log}, {api_responses[0].model_dump()}"
        assert final_response.parsed_outputs is not None, "parsed_outputs is None"
    assert len(api_responses) == 1, "api_count is not 1"
    assert (
        getattr(final_response.api_response.choices[0].message, "function_call", None)
        is None
    ), "function_call is not None"

    assert len([c for c in chunks if c.function_call is not None]) == 0

    assert api_responses[0].choices[0].message.content is not None, "content is None"
