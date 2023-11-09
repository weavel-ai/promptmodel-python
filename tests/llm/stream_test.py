import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.types import LLMResponse, LLMStreamResponse


def test_stream(mocker):
    llm = LLM()
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce yourself in 50 words."},
    ]

    stream_res: Generator[LLMStreamResponse, None, None] = llm.stream(
        messages=test_messages,
        functions=[],
        model="gpt-3.5-turbo",
    )
    error_count = 0
    api_responses: List[ModelResponse] = []
    for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"

    assert (
        api_responses[0].choices[0]["message"]["content"] is not None
    ), "content is None"
    assert api_responses[0]["response_ms"] is not None, "response_ms is None"

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


def test_stream_with_functions(mocker):
    # JSON Schema to pass to OpenAI
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: Generator[LLMStreamResponse, None, None] = llm.stream(
        messages=messages,
        functions=functions,
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
async def test_astream(mocker):
    llm = LLM()
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce yourself in 50 words."},
    ]

    stream_res: Generator[LLMStreamResponse, None, None] = llm.astream(
        messages=test_messages,
        functions=[],
        model="gpt-3.5-turbo",
    )
    error_count = 0
    api_responses: List[ModelResponse] = []
    async for res in stream_res:
        if res.error:
            error_count += 1
            print("ERROR")
            print(res.error)
            print(res.error_log)
        if res.api_response:
            api_responses.append(res.api_response)

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"

    assert (
        api_responses[0].choices[0]["message"]["content"] is not None
    ), "content is None"
    assert api_responses[0]["response_ms"] is not None, "response_ms is None"

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


@pytest.mark.asyncio
async def test_astream_with_functions(mocker):
    # JSON Schema to pass to OpenAI
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    llm = LLM()
    stream_res: Generator[LLMStreamResponse, None, None] = llm.astream(
        messages=messages,
        functions=functions,
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
