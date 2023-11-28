import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import nest_asyncio
from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import LLMResponse, LLMStreamResponse
from promptmodel.utils.async_utils import run_async_in_sync


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
        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)
    print(api_responses)
    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"

    assert api_responses[0].choices[0].message.content is not None, "content is None"
    assert api_responses[0]._response_ms is not None, "response_ms is None"

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
async def test_astream(mocker):
    llm = LLM()
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce yourself in 50 words."},
    ]

    stream_res: AsyncGenerator[LLMStreamResponse, None] = llm.astream(
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
        if (
            res.api_response
            and getattr(res.api_response.choices[0], "delta", None) is None
        ):
            api_responses.append(res.api_response)

    assert error_count == 0, "error_count is not 0"
    assert len(api_responses) == 1, "api_count is not 1"

    assert api_responses[0].choices[0].message.content is not None, "content is None"
    assert api_responses[0]._response_ms is not None, "response_ms is None"

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

    api_response_dict = api_responses[0].model_dump()
    api_response_dict.update({"response_ms": api_responses[0]._response_ms})

    assert (
        kwargs["json"]["api_response"] == api_response_dict
    ), "api_response is not equal"
