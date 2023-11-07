import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any
from litellm import ModelResponse

from promptmodel.apis.base import APIClient, AsyncAPIClient
from promptmodel.utils.crypto import decrypt_message, generate_api_key, encrypt_message


@pytest.mark.asyncio
async def test_get_headers(mocker):
    api_client = APIClient()
    async_api_client = AsyncAPIClient()

    mock_exit = mocker.patch("builtins.exit", side_effect=SystemExit)
    api_key = generate_api_key()
    mock_envirion_get = MagicMock(return_value=api_key)

    gt_cli_output = {
        "Authorization": f"Bearer {decrypt_message(encrypt_message(api_key))}"
    }
    gt_api_output = {"Authorization": f"Bearer {api_key}"}
    # No user
    mocker.patch("promptmodel.apis.base.read_config", return_value={})
    with pytest.raises(SystemExit):
        res = api_client._get_headers()
    mock_exit.assert_called_once()
    mock_exit.reset_mock()

    with pytest.raises(SystemExit):
        res = await async_api_client._get_headers()
    mock_exit.assert_called_once()
    mock_exit.reset_mock()

    mocker.patch("promptmodel.apis.base.read_config", return_value={"user": {}})

    with pytest.raises(Exception):
        res = api_client._get_headers()
    with pytest.raises(Exception):
        res = await async_api_client._get_headers()

    mocker.patch(
        "promptmodel.apis.base.read_config",
        return_value={"user": {"encrypted_api_key": encrypt_message(api_key)}},
    )
    res = api_client._get_headers()
    assert res == gt_cli_output, "API key is not decrypted properly"
    res = await async_api_client._get_headers()
    assert res == gt_api_output, "API key is not decrypted properly"

    mocker.patch("promptmodel.apis.base.os.environ.get", mock_envirion_get)
    res = api_client._get_headers(use_cli_key=False)
    assert res == gt_api_output, "API key is not retrieved properly"
    res = await async_api_client._get_headers(use_cli_key=False)
    assert res == gt_api_output, "API key is not retrieved properly"


@pytest.mark.asyncio
async def test_execute(mocker):
    api_client = APIClient()
    async_api_client = AsyncAPIClient()
    mock_exit = mocker.patch("builtins.exit", side_effect=SystemExit)
    mocker.patch("promptmodel.apis.base.APIClient._get_headers", return_value={})
    mocker.patch("promptmodel.apis.base.AsyncAPIClient._get_headers", return_value={})

    mock_request = mocker.patch(
        "promptmodel.apis.base.requests.request", return_value=None
    )
    mock_async_request = mocker.patch(
        "promptmodel.apis.base.httpx.AsyncClient.request", return_value=None
    )
    with pytest.raises(SystemExit):
        res = api_client.execute(path="test")
    mock_request.assert_called_once()
    mock_request.reset_mock()
    mock_exit.assert_called_once()
    mock_exit.reset_mock()

    res = await async_api_client.execute(path="test")
    mock_async_request.assert_called_once()
    mock_async_request.reset_mock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_request = mocker.patch(
        "promptmodel.apis.base.requests.request", return_value=mock_response
    )
    mock_async_request = mocker.patch(
        "promptmodel.apis.base.httpx.AsyncClient.request", return_value=mock_response
    )

    res = api_client.execute(path="test")
    mock_request.assert_called_once()
    assert res == mock_response, "Response is not returned properly"
    mock_request.reset_mock()

    res = await async_api_client.execute(path="test")
    mock_async_request.assert_called_once()
    assert res == mock_response, "Response is not returned properly"
    mock_async_request.reset_mock()

    mock_response.status_code = 403
    mock_request = mocker.patch(
        "promptmodel.apis.base.requests.request", return_value=mock_response
    )
    mock_async_request = mocker.patch(
        "promptmodel.apis.base.httpx.AsyncClient.request", return_value=mock_response
    )
    with pytest.raises(SystemExit):
        res = api_client.execute(path="test")
    mock_request.assert_called_once()
    mock_request.reset_mock()
    mock_exit.assert_called_once()
    mock_exit.reset_mock()

    res = await async_api_client.execute(path="test")
    mock_async_request.assert_called_once()
    mock_async_request.reset_mock()

    mock_response.status_code = 500
    mock_request = mocker.patch(
        "promptmodel.apis.base.requests.request", return_value=mock_response
    )
    mock_async_request = mocker.patch(
        "promptmodel.apis.base.httpx.AsyncClient.request", return_value=mock_response
    )
    with pytest.raises(SystemExit):
        res = api_client.execute(path="test")
    mock_request.assert_called_once()
    mock_request.reset_mock()
    mock_exit.assert_called_once()
    mock_exit.reset_mock()

    res = await async_api_client.execute(path="test")
    mock_async_request.assert_called_once()
    mock_async_request.reset_mock()
