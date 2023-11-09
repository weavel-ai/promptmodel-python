import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import asyncio
from typing import Optional
from uuid import uuid4
from dataclasses import dataclass
from websockets.exceptions import ConnectionClosedOK
from promptmodel.websocket.websocket_client import DevWebsocketClient
from promptmodel.utils.enums import LocalTask, ParsingType
from promptmodel.database.models import PromptModelVersion
from promptmodel.utils.types import FunctionSchema


@dataclass
class PromptModelInterface:
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
    websocket_client._devapp.prompt_models = [PromptModelInterface("test_module")]
    websocket_client._devapp.samples = {
        "sample_1": {"user_message": "What is the weather like in Boston?"}
    }
    websocket_client._devapp.functions = {
        "get_current_weather": {
            "schema": FunctionSchema(**get_current_weather_desc),
            "function": get_current_weather,
        }
    }
    prompt_model_uuid = uuid4()
    param = {
        "prompt_model_uuid": prompt_model_uuid,
        "uuid": uuid4(),
        "status": "broken",
        "from_uuid": None,
        "model": "gpt-3.5-turbo",
        "parsing_type": None,
        "output_keys": None,
        "functions": [],
    }
    mock_version = PromptModelVersion(**param)
    mocker.patch(
        "promptmodel.websocket.websocket_client.get_sample_input",
        new_callable=MagicMock,
        return_value={
            "name": "sample_1",
            "contents": {"user_message": "What is the weather like in Boston?"},
        },
    )
    mocker.patch(
        "promptmodel.websocket.websocket_client.get_prompt_model_uuid",
        new_callable=MagicMock,
        return_value={"uuid": prompt_model_uuid},
    )
    mocker.patch(
        "promptmodel.websocket.websocket_client.create_prompt_model_version",
        new_callable=MagicMock,
        return_value=mock_version,
    )
    mocker.patch(
        "promptmodel.websocket.websocket_client.create_prompt", new_callable=MagicMock
    )
    mocker.patch(
        "promptmodel.websocket.websocket_client.update_prompt_model_version",
        new_callable=MagicMock,
    )
    create_run_log_mock = mocker.patch(
        "promptmodel.websocket.websocket_client.create_run_log", new_callable=MagicMock
    )

    # success case
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_PROMPT_MODEL,
            "prompt_model_name": "test_module",
            "sample_name": "sample_1",
            "prompts": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                    "step": 1,
                },
                {"role": "user", "content": "{user_message}", "step": 2},
            ],
            "model": "gpt-3.5-turbo",
            "uuid": None,
            "from_uuid": None,
            "parsing_type": None,
            "output_keys": None,
            "functions": ["get_current_weather"],
        },
        ws=mock_websocket,
    )
    create_run_log_mock.assert_called_once()
    _, kwargs = create_run_log_mock.call_args
    assert kwargs["function_call"] is not None, "function_call is None"
    create_run_log_mock.reset_mock()
    print(
        "======================================================================================="
    )

    # success case with no function call
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_PROMPT_MODEL,
            "prompt_model_name": "test_module",
            "sample_name": "sample_1",
            "prompts": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                    "step": 1,
                },
                {"role": "user", "content": "{user_message}", "step": 2},
            ],
            "model": "gpt-3.5-turbo",
            "uuid": None,
            "from_uuid": None,
            "parsing_type": None,
            "output_keys": None,
            "functions": [],
        },
        ws=mock_websocket,
    )
    create_run_log_mock.assert_called_once()
    _, kwargs = create_run_log_mock.call_args
    assert kwargs["function_call"] is None, "function_call is not None"
    create_run_log_mock.reset_mock()
    print(
        "======================================================================================="
    )

    # success case with parsing
    system_prompt_with_format = """
You are a helpful assistant.

This is your output format. Keep the string between < type=< >>, </ > as it is.
<temperature type=<float>>
(value here)
</temperature>
    """
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_PROMPT_MODEL,
            "prompt_model_name": "test_module",
            "sample_name": "sample_1",
            "prompts": [
                {"role": "system", "content": system_prompt_with_format, "step": 1},
                {"role": "user", "content": "{user_message}", "step": 2},
            ],
            "model": "gpt-3.5-turbo",
            "uuid": None,
            "from_uuid": None,
            "parsing_type": ParsingType.HTML.value,
            "output_keys": ["temperature"],
            "functions": ["get_current_weather"],
        },
        ws=mock_websocket,
    )
    create_run_log_mock.assert_called_once()
    _, kwargs = create_run_log_mock.call_args
    assert kwargs["function_call"] is not None, "function_call is None"
    create_run_log_mock.reset_mock()
    print(
        "======================================================================================="
    )

    # early failed case due to function not existing
    await websocket_client._DevWebsocketClient__handle_message(
        message={
            "type": LocalTask.RUN_PROMPT_MODEL,
            "prompt_model_name": "test_module",
            "sample_name": "sample_1",
            "prompts": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                    "step": 1,
                },
                {"role": "user", "content": "{user_message}", "step": 2},
            ],
            "model": "gpt-3.5-turbo",
            "uuid": None,
            "from_uuid": None,
            "parsing_type": None,
            "output_keys": None,
            "functions": ["get_weather"],
        },
        ws=mock_websocket,
    )
    create_run_log_mock.assert_not_called()
