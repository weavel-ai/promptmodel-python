import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from promptmodel.llms.llm import LLM, ParseResult
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import LLMResponse, LLMStreamResponse
from promptmodel.types.enums import ParsingType

proxy = LLMProxy(name="test")


def test_run(mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: LLMResponse = proxy.run({})
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()
    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None


@pytest.mark.asyncio
async def test_arun(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: LLMResponse = await proxy.arun({})
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()
    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None


def test_stream(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: Generator[LLMStreamResponse, None, None] = proxy.stream({})
    chunks: List[LLMStreamResponse] = []
    for chunk in res:
        chunks.append(chunk)
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()

    assert chunks[-1].api_response is not None
    assert len([chunk for chunk in chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.parsed_outputs is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.raw_output is not None]) > 0


@pytest.mark.asyncio
async def test_astream(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: AsyncGenerator[LLMStreamResponse, None] = proxy.astream({})
    chunks: List[LLMStreamResponse] = []
    async for chunk in res:
        chunks.append(chunk)
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()

    assert chunks[-1].api_response is not None
    assert len([chunk for chunk in chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.parsed_outputs is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.raw_output is not None]) > 0


def test_run_and_parse(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: LLMResponse = proxy.run({})
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()
    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None

    fetch_prompts.reset_mock()
    async_log_to_cloud.reset_mock()

    # mock run
    mock_run = MagicMock()
    mock_run.return_value = LLMResponse(
        api_response=res.api_response, parsed_outputs={"key": "value"}
    )
    mocker.patch("promptmodel.llms.llm.LLM.run", mock_run)
    mock_res = proxy.run({})
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()
    assert mock_res.raw_output is None
    assert mock_res.error is None or res.error is False
    assert mock_res.api_response is not None
    assert mock_res.parsed_outputs is not None


@pytest.mark.asyncio
async def test_arun_and_parse(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: LLMResponse = await proxy.arun({})
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()
    assert res.raw_output is not None
    assert res.error is None or res.error is False
    assert res.api_response is not None
    assert res.parsed_outputs is None

    fetch_prompts.reset_mock()
    async_log_to_cloud.reset_mock()

    # mock run
    mock_run = AsyncMock()
    mock_run.return_value = LLMResponse(
        api_response=res.api_response, parsed_outputs={"key": "value"}
    )
    mocker.patch("promptmodel.llms.llm.LLM.arun", mock_run)
    mock_res: LLMResponse = await proxy.arun({})
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()
    assert mock_res.raw_output is None
    assert mock_res.error is None or res.error is False
    assert mock_res.api_response is not None
    assert mock_res.parsed_outputs is not None


def test_stream_and_parse(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: Generator[LLMStreamResponse, None, None] = proxy.stream({})
    chunks: List[LLMStreamResponse] = []
    for chunk in res:
        chunks.append(chunk)
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()

    assert chunks[-1].api_response is not None
    assert len([chunk for chunk in chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.parsed_outputs is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.raw_output is not None]) > 0

    fetch_prompts.reset_mock()
    async_log_to_cloud.reset_mock()

    def mock_stream_generator(*args, **kwargs):
        yield LLMStreamResponse(parsed_outputs={"key": "value"})
        yield LLMStreamResponse(api_response=chunks[-1].api_response)

    mock_run = MagicMock(side_effect=mock_stream_generator)
    mocker.patch("promptmodel.llms.llm.LLM.stream", mock_run)

    mock_res: Generator[LLMStreamResponse, None, None] = proxy.stream({})
    mock_chunks: List[LLMStreamResponse] = []
    for chunk in mock_res:
        mock_chunks.append(chunk)
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()

    assert mock_chunks[-1].api_response is not None
    assert len([chunk for chunk in mock_chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in mock_chunks if chunk.parsed_outputs is not None]) > 0
    assert len([chunk for chunk in mock_chunks if chunk.raw_output is not None]) == 0


@pytest.mark.asyncio
async def test_astream_and_parse(
    mocker, mock_fetch_prompts: AsyncMock, mock_async_log_to_cloud: AsyncMock
):
    fetch_prompts = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy.fetch_prompts", mock_fetch_prompts
    )
    async_log_to_cloud = mocker.patch(
        "promptmodel.llms.llm_proxy.LLMProxy._async_log_to_cloud",
        mock_async_log_to_cloud,
    )

    res: AsyncGenerator[LLMStreamResponse, None] = proxy.astream({})
    chunks: List[LLMStreamResponse] = []
    async for chunk in res:
        chunks.append(chunk)
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()

    assert chunks[-1].api_response is not None
    assert len([chunk for chunk in chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.parsed_outputs is not None]) == 0
    assert len([chunk for chunk in chunks if chunk.raw_output is not None]) > 0

    fetch_prompts.reset_mock()
    async_log_to_cloud.reset_mock()

    async def mock_stream_generator(*args, **kwargs):
        yield LLMStreamResponse(parsed_outputs={"key": "value"})
        yield LLMStreamResponse(api_response=chunks[-1].api_response)

    mock_run = MagicMock(side_effect=mock_stream_generator)
    mocker.patch("promptmodel.llms.llm.LLM.astream", mock_run)

    mock_res: AsyncGenerator[LLMStreamResponse, None] = proxy.astream({})
    mock_chunks: List[LLMStreamResponse] = []
    async for chunk in mock_res:
        mock_chunks.append(chunk)
    fetch_prompts.assert_called_once()
    async_log_to_cloud.assert_called_once()

    assert mock_chunks[-1].api_response is not None
    assert len([chunk for chunk in mock_chunks if chunk.error is not None]) == 0
    assert len([chunk for chunk in mock_chunks if chunk.parsed_outputs is not None]) > 0
    assert len([chunk for chunk in mock_chunks if chunk.raw_output is not None]) == 0
