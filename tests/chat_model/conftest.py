import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from promptmodel.llms.llm_proxy import LLMProxy


async def echo_coroutine(*args, **kwargs):
    # print(args, kwargs)
    return args, kwargs


@pytest.fixture
def mock_fetch_chat_log():
    mock_fetch_chat_log = AsyncMock()
    mock_chat_log = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "session_uuid": "testuuid",
        },
        {"role": "user", "content": "Hello!", "session_uuid": "testuuid"},
        {
            "role": "assistant",
            "content": "Hello! How can I help you?",
            "session_uuid": "testuuid",
        },
    ]

    mock_fetch_chat_log.return_value = mock_chat_log

    return mock_fetch_chat_log


@pytest.fixture
def mock_fetch_chat_model():
    mock_fetch_chat_model = AsyncMock()
    mock_instruction = [
        {"role": "system", "content": "You are a helpful assistant.", "step": 1},
    ]
    mock_version_details = {
        "model": "gpt-3.5-turbo",
        "uuid": "testuuid",
    }
    mock_fetch_chat_model.return_value = (mock_instruction, mock_version_details)

    return mock_fetch_chat_model


@pytest.fixture
def mock_async_chat_log_to_cloud():
    mock_async_chat_log_to_cloud = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_async_chat_log_to_cloud.return_value = mock_response

    return mock_async_chat_log_to_cloud
