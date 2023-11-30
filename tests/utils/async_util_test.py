import pytest
import asyncio
import nest_asyncio
from typing import Coroutine
from unittest.mock import AsyncMock

from promptmodel.utils.async_utils import run_async_in_sync

nest_asyncio.apply()


def test_sync_context(mocker):
    coro = AsyncMock(return_value="test")
    res = run_async_in_sync(coro())
    coro.assert_called_once()

    assert res == "test", "res is not test"


def test_sync_async_context(mocker):
    coro = AsyncMock(return_value="test")

    async def async_context(coro: Coroutine):
        res = await coro
        return res

    res = asyncio.run(async_context(coro()))
    coro.assert_called_once()

    assert res == "test", "res is not test"


@pytest.mark.asyncio
async def test_async_context(mocker):
    coro = AsyncMock(return_value="test")
    res = await coro()
    coro.assert_called_once()

    assert res == "test", "res is not test"


@pytest.mark.asyncio
async def test_async_sync_context(mocker):
    coro = AsyncMock(return_value="test")
    print("ready")

    def sync_context(coro: Coroutine):
        return run_async_in_sync(coro)

    res = sync_context(coro())
    coro.assert_called_once()

    assert res == "test", "res is not test"
