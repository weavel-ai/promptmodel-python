import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from promptmodel.llms.llm_proxy import LLMProxy


async def echo_coroutine(*args, **kwargs):
    # print(args, kwargs)
    return args, kwargs


@pytest.fixture
def mock_fetch_prompts():
    mock_fetch_prompts = AsyncMock()
    mock_prompts = [
        {"role": "system", "content": "You are a helpful assistant.", "step": 1},
        {"role": "user", "content": "Hello!", "step": 2},
    ]
    mock_version_details = {
        "model": "gpt-3.5-turbo",
        "uuid": "testuuid",
        "parsing_type": None,
        "output_keys": None,
    }
    mock_fetch_prompts.return_value = (mock_prompts, mock_version_details)

    return mock_fetch_prompts


@pytest.fixture
def mock_async_log_to_cloud():
    mock_async_log_to_cloud = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_async_log_to_cloud.return_value = mock_response

    return mock_async_log_to_cloud
