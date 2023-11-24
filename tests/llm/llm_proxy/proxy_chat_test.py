import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from promptmodel.llms.llm import LLM, ParseResult
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import LLMResponse, LLMStreamResponse
from promptmodel.types.enums import ParsingType

proxy = LLMProxy(name="test")


def test_chat_run(
    mocker,
    mock_fetch_chat_model: AsyncMock,
    mock_async_chat_log_to_cloud: AsyncMock,
):
    fetch_chat_model = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_chat_model", mock_fetch_chat_model
    )

    async_chat_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_chat_log_to_cloud",
        mock_async_chat_log_to_cloud,
    )

    res: LLMResponse = proxy.chat_run(session_uuid="testuuid")

    fetch_chat_model.assert_called_once()
    async_chat_log_to_cloud.assert_called_once()

    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None
    if isinstance(res.api_response.usage, dict):
        assert res.api_response.usage["prompt_tokens"] > 15
        print(res.api_response.usage["prompt_tokens"])
    else:
        assert res.api_response.usage.prompt_tokens > 15
        print(res.api_response.usage.prompt_tokens)


@pytest.mark.asyncio
async def test_chat_arun(
    mocker,
    mock_fetch_chat_model: AsyncMock,
    mock_async_chat_log_to_cloud: AsyncMock,
):
    fetch_chat_model = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_chat_model", mock_fetch_chat_model
    )

    async_chat_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_chat_log_to_cloud",
        mock_async_chat_log_to_cloud,
    )

    res: LLMResponse = await proxy.chat_arun(session_uuid="testuuid")

    fetch_chat_model.assert_called_once()
    async_chat_log_to_cloud.assert_called_once()

    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None


def test_chat_stream(
    mocker,
    mock_fetch_chat_model: AsyncMock,
    mock_async_chat_log_to_cloud: AsyncMock,
):
    fetch_chat_model = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_chat_model", mock_fetch_chat_model
    )

    async_chat_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_chat_log_to_cloud",
        mock_async_chat_log_to_cloud,
    )

    res: Generator[LLMStreamResponse, None, None] = proxy.chat_stream(
        session_uuid="testuuid"
    )
    chunks: List[LLMStreamResponse] = []
    for chunk in res:
        chunks.append(chunk)

    fetch_chat_model.assert_called_once()
    async_chat_log_to_cloud.assert_called_once()

    assert chunks[-1].api_response is not None
    assert len([chunk for chunk in chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.parsed_outputs is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.raw_output is not None]) > 0


@pytest.mark.asyncio
async def test_chat_astream(
    mocker,
    mock_fetch_chat_model: AsyncMock,
    mock_async_chat_log_to_cloud: AsyncMock,
):
    fetch_chat_model = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_chat_model", mock_fetch_chat_model
    )

    async_chat_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_chat_log_to_cloud",
        mock_async_chat_log_to_cloud,
    )

    res: AsyncGenerator[LLMStreamResponse, None] = proxy.chat_astream(
        session_uuid="testuuid"
    )
    chunks: List[LLMStreamResponse] = []
    async for chunk in res:
        chunks.append(chunk)

    fetch_chat_model.assert_called_once()
    async_chat_log_to_cloud.assert_called_once()

    assert chunks[-1].api_response is not None
    assert len([chunk for chunk in chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.parsed_outputs is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.raw_output is not None]) > 0


def test_chat_run_extra_long_input(
    mocker,
    mock_fetch_chat_model: AsyncMock,
    mock_async_chat_log_to_cloud: AsyncMock,
):
    mocker.patch("promptmodel.llms.llm_proxy.get_max_tokens", return_value=10)

    fetch_chat_model = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_chat_model", mock_fetch_chat_model
    )

    async_chat_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_chat_log_to_cloud",
        mock_async_chat_log_to_cloud,
    )

    res: LLMResponse = proxy.chat_run(session_uuid="testuuid")

    fetch_chat_model.assert_called_once()
    async_chat_log_to_cloud.assert_called_once()

    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None
    if isinstance(res.api_response.usage, dict):
        assert res.api_response.usage["prompt_tokens"] < 15
        print(res.api_response.usage["prompt_tokens"])
    else:
        assert res.api_response.usage.prompt_tokens < 15
        print(res.api_response.usage.prompt_tokens)
