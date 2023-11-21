import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import asyncio
from typing import Optional
from uuid import uuid4
from dataclasses import dataclass
from websockets.exceptions import ConnectionClosedOK
from promptmodel.websocket.websocket_client import DevWebsocketClient
from promptmodel.types.enums import LocalTask, ParsingType
from promptmodel.database.models import (
    ChatModelVersion,
    ChatModel,
    ChatLog,
    ChatLogSession,
)
from promptmodel.types.response import FunctionSchema


@dataclass
class ChatModelInterface:
    name: str
    default_model: str = "gpt-3.5-turbo"


def get_current_weather(location: str, unit: Optional[str] = "celsius"):
    return "13 degrees celsius"


get_current_weather_desc = {
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


@pytest.mark.asyncio
async def test_run_model_function_call(
    mocker, websocket_client: DevWebsocketClient, mock_websocket: AsyncMock
):
    websocket_client._devapp.chat_models = [ChatModelInterface("test_module")]
    websocket_client._devapp.samples = {
        "sample_1": {"user_message": "What is the weather like in Boston?"}
    }
    websocket_client._devapp.functions = {
        "get_current_weather": {
            "schema": FunctionSchema(**get_current_weather_desc),
            "function": get_current_weather,
        }
    }
    chat_model_uuid = uuid4()
    param = {
        "chat_model_uuid": chat_model_uuid,
        "uuid": uuid4(),
        "status": "broken",
        "from_uuid": None,
        "model": "gpt-3.5-turbo",
        "system_prompt": {"role": "system", "content": "You are a helpful assistant."},
        "functions": [],
    }
    mock_version = ChatModelVersion(**param)

    mocker.patch(
        "promptmodel.websocket.websocket_client.ChatModel.get",
        new_callable=MagicMock,
        return_value=ChatModel(**{"uuid": chat_model_uuid}),
    )

    create_chat_model_version_mock = mocker.patch(
        "promptmodel.websocket.websocket_client.ChatModelVersion.create",
        new_callable=MagicMock,
        return_value=mock_version,
    )

    mocker.patch(
        "promptmodel.websocket.websocket_client.ChatModelVersion.update",
        new_callable=MagicMock,
    )

    # ready for return value
    mock_return_value = ChatLog(role="system", content="You are a helpful assistant.")

    where_mock = MagicMock()
    order_mock = MagicMock()

    where_mock.where.return_value = order_mock
    order_mock.order_by.return_value = mock_return_value

    # `ChatLog.select` mocking
    select_chat_log_mock = mocker.patch(
        "promptmodel.websocket.websocket_client.ChatLog.select", return_value=where_mock
    )

    create_chat_log_session_mock = mocker.patch(
        "promptmodel.websocket.websocket_client.ChatLogSession.create",
        new_callable=MagicMock,
    )

    create_chat_log_mock = mocker.patch(
        "promptmodel.websocket.websocket_client.ChatLog.create", new_callable=MagicMock
    )

    insert_many_chat_log_mock = mocker.patch(
        "promptmodel.websocket.websocket_client.ChatLog.insert_many",
        new_callable=MagicMock,
    )

    # success case without session_uuid
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "session_uuid": None,
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "new_messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Boston?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "uuid": None,
            "from_uuid": None,
            "functions": ["get_current_weather"],
        },
        ws=mock_websocket,
    )

    # if session_uuid is None,
    # call ChatLogSession.create()
    create_chat_log_session_mock.assert_called_once()
    create_chat_model_version_mock.assert_called_once()
    # call chatLog.create 4 times : system, function_call, function_response, assistant output
    assert create_chat_log_mock.call_count == 4
    insert_many_chat_log_mock.assert_called_once()

    create_chat_log_mock.reset_mock()
    insert_many_chat_log_mock.reset_mock()

    # success case with function_call
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "session_uuid": "testuuid",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "new_messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Boston?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "uuid": "test_uuid",
            "from_uuid": None,
            "functions": ["get_current_weather"],
        },
        ws=mock_websocket,
    )
    select_chat_log_mock.assert_called_once()
    # call chatLog.create 5 times : function_call, function_response, assistant output
    assert create_chat_log_mock.call_count == 3
    insert_many_chat_log_mock.assert_called_once()
    # _, kwargs = create_chat_log_mock.call_args
    # assert kwargs["tool_calls"] is not None, "tool_calls is None"
    create_chat_log_mock.reset_mock()
    select_chat_log_mock.reset_mock()
    insert_many_chat_log_mock.reset_mock()
    print(
        "======================================================================================="
    )

    # success case with no function call
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "session_uuid": "testuuid",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "new_messages": [
                {
                    "role": "user",
                    "content": "Hello?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "uuid": "test_uuid",
            "from_uuid": None,
            "functions": [],
        },
        ws=mock_websocket,
    )
    select_chat_log_mock.assert_called_once()
    # call chatLog.create 1 times : assistant output
    assert create_chat_log_mock.call_count == 1
    insert_many_chat_log_mock.assert_called_once()
    create_chat_log_mock.reset_mock()
    select_chat_log_mock.reset_mock()
    insert_many_chat_log_mock.reset_mock()
    print(
        "======================================================================================="
    )

    # early failed case due to function not existing
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "session_uuid": "testuuid",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "new_messages": [
                {
                    "role": "user",
                    "content": "Hello?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "uuid": "test_uuid",
            "from_uuid": None,
            "functions": ["get_weather"],
        },
        ws=mock_websocket,
    )
    create_chat_log_mock.assert_not_called()
