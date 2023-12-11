import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from websockets.exceptions import ConnectionClosedOK
from promptmodel.websocket.websocket_client import DevWebsocketClient
from promptmodel.types.enums import LocalTask


@pytest.mark.asyncio
async def test_connect_to_gateway(
    mocker, websocket_client: DevWebsocketClient, mock_websocket: AsyncMock
):
    project_uuid = "test_uuid"
    connection_name = "test_project"
    cli_access_header = {"Authorization": "Bearer testtoken"}

    with patch.object(
        websocket_client, "_DevWebsocketClient__handle_message", new_callable=AsyncMock
    ) as mock_function:
        mock_function.side_effect = ConnectionClosedOK(None, None)
        with patch(
            "promptmodel.websocket.websocket_client.connect",
            new_callable=MagicMock,
            return_value=mock_websocket,
        ) as mock_connect:
            # 5초 후에 자동으로 테스트 종료
            await websocket_client.connect_to_gateway(
                project_uuid, connection_name, cli_access_header, retries=1
            ),
            mock_connect.assert_called_once()
            mock_websocket.recv.assert_called_once()
            websocket_client._DevWebsocketClient__handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_local_tasks(
    mocker, websocket_client: DevWebsocketClient, mock_websocket: AsyncMock
):
    websocket_client._devapp.function_models = {}
    websocket_client._devapp.samples = {}
    websocket_client._devapp.functions = {"test_function": "test_function"}

    await websocket_client._DevWebsocketClient__handle_message(
        message={"type": LocalTask.LIST_CODE_FUNCTIONS}, ws=mock_websocket
    )
