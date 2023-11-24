import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import asyncio
from typing import Optional
from uuid import uuid4
from dataclasses import dataclass
from websockets.exceptions import ConnectionClosedOK
from promptmodel.websocket.websocket_client import DevWebsocketClient
from promptmodel.types.enums import LocalTask, LocalTaskErrorType
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
    mocker,
    websocket_client: DevWebsocketClient,
    mock_websocket: AsyncMock,
    mock_json_dumps: MagicMock,
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

    function_schemas_in_db = [FunctionSchema(**get_current_weather_desc).model_dump()]

    function_schemas_in_db[0]["mock_response"] = "13"

    mocker.patch("promptmodel.websocket.websocket_client.json.dumps", mock_json_dumps)

    # success case with function_call
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "old_messages": [],
            "new_messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Boston?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "functions": ["get_current_weather"],
            "function_schemas": function_schemas_in_db,
        },
        ws=mock_websocket,
    )

    call_args_list = mock_websocket.send.call_args_list
    data = [arg.args[0] for arg in call_args_list]

    assert len([d for d in data if d["status"] == "failed"]) == 0
    assert len([d for d in data if d["status"] == "completed"]) == 1

    assert len([d for d in data if "function_response" in d]) == 1
    assert [d for d in data if "function_response" in d][0]["function_response"][
        "name"
    ] == "get_current_weather"

    assert len([d for d in data if "function_call" in d]) > 1
    assert len([d for d in data if "raw_output" in d]) > 0

    mock_websocket.send.reset_mock()
    function_schemas_in_db[0]["mock_response"] = "13"

    print(
        "======================================================================================="
    )

    # success case with no function call
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "old_messages": [],
            "new_messages": [
                {
                    "role": "user",
                    "content": "Hello?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "functions": [],
            "function_schemas": [],
        },
        ws=mock_websocket,
    )

    call_args_list = mock_websocket.send.call_args_list
    data = [arg.args[0] for arg in call_args_list]

    assert len([d for d in data if d["status"] == "failed"]) == 0
    assert len([d for d in data if d["status"] == "completed"]) == 1

    assert len([d for d in data if "function_response" in d]) == 0

    assert len([d for d in data if "function_call" in d]) == 0
    assert len([d for d in data if "raw_output" in d]) > 0

    mock_websocket.send.reset_mock()
    function_schemas_in_db[0]["mock_response"] = "13"

    print(
        "======================================================================================="
    )

    # FUNCTION_CALL_FAILED_ERROR case
    def error_raise_function(*args, **kwargs):
        raise Exception("error")

    websocket_client._devapp.functions = {
        "get_current_weather": {
            "schema": FunctionSchema(**get_current_weather_desc),
            "function": error_raise_function,
        }
    }

    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "old_messages": [],
            "new_messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Boston?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "functions": ["get_current_weather"],
            "function_schemas": function_schemas_in_db,
        },
        ws=mock_websocket,
    )
    call_args_list = mock_websocket.send.call_args_list
    # print(call_args_list)
    data = [arg.args[0] for arg in call_args_list]

    assert len([d for d in data if d["status"] == "failed"]) == 1
    assert [d for d in data if d["status"] == "failed"][0][
        "error_type"
    ] == LocalTaskErrorType.FUNCTION_CALL_FAILED_ERROR.value
    assert len([d for d in data if d["status"] == "completed"]) == 0
    assert len([d for d in data if "function_response" in d]) == 0
    assert len([d for d in data if "function_call" in d]) > 1
    assert len([d for d in data if "raw_output" in d]) == 0
    mock_websocket.send.reset_mock()
    function_schemas_in_db[0]["mock_response"] = "13"

    # function not in code case, should use mock_response
    websocket_client._devapp.functions = {}
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_CHAT_MODEL,
            "chat_model_name": "test_module",
            "system_prompt": {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            "old_messages": [],
            "new_messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Boston?",
                }
            ],
            "model": "gpt-3.5-turbo",
            "functions": ["get_weather"],
            "function_schemas": function_schemas_in_db,
        },
        ws=mock_websocket,
    )
    call_args_list = mock_websocket.send.call_args_list
    print(call_args_list)
    data = [arg.args[0] for arg in call_args_list]

    assert len([d for d in data if d["status"] == "failed"]) == 0
    assert len([d for d in data if d["status"] == "completed"]) == 1

    assert len([d for d in data if "function_response" in d]) == 1
    assert (
        "FAKE RESPONSE"
        in [d for d in data if "function_response" in d][0]["function_response"][
            "response"
        ]
    )
    assert len([d for d in data if "function_call" in d]) > 1
    assert len([d for d in data if "raw_output" in d]) > 0
