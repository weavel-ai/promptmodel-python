import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import ChatModelConfig


async def echo_coroutine(*args, **kwargs):
    # print(args, kwargs)
    return args, kwargs


@pytest.fixture
def mock_fetch_chat_model():
    mock_fetch_chat_model = AsyncMock()
    mock_instruction = "You are a helpful assistant."
    mock_version_details = {
        "model": "gpt-4-1106-preview",
        "uuid": "testuuid",
    }
    mock_message_logs = [
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

    mock_fetch_chat_model.return_value = (
        mock_instruction,
        mock_version_details,
        mock_message_logs,
    )

    return mock_fetch_chat_model


@pytest.fixture
def mock_async_chat_log_to_cloud():
    mock_async_chat_log_to_cloud = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_async_chat_log_to_cloud.return_value = mock_response

    return mock_async_chat_log_to_cloud


@pytest.fixture
def mock_async_make_session_cloud():
    mock_async_make_session_cloud = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_async_make_session_cloud.return_value = mock_response

    return mock_async_make_session_cloud
