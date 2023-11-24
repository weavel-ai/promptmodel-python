import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from promptmodel.websocket.websocket_client import DevWebsocketClient
from promptmodel.dev_app import DevApp


async def echo_coroutine(*args, **kwargs):
    # print(args, kwargs)
    return args, kwargs


@pytest.fixture
def websocket_client():
    websocket_client = DevWebsocketClient(_devapp=DevApp())
    return websocket_client


@pytest.fixture
def mock_websocket():
    # 모의 WebSocketClientProtocol 객체 생성
    mock_websocket = AsyncMock()

    async def aenter(self):
        return self

    async def aexit(self, exc_type, exc_value, traceback):
        pass

    mock_websocket.__aenter__ = aenter
    mock_websocket.__aexit__ = aexit
    mock_websocket.recv = AsyncMock(return_value='{"key" : "value"}')
    mock_websocket.send = AsyncMock()

    return mock_websocket


@pytest.fixture
def mock_json_dumps():
    mock_json_dumps = MagicMock()
    # it should return exactly the same as the input
    mock_json_dumps.side_effect = lambda data, *args, **kwargs: data
    return mock_json_dumps
