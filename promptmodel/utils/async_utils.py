import asyncio
from typing import Coroutine


def run_async_in_sync(coro: Coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        # loop.close()
        return result

    return loop.run_until_complete(coro)


def run_async_in_sync_threadsafe(coro: Coroutine, main_loop: asyncio.AbstractEventLoop):
    future = asyncio.run_coroutine_threadsafe(coro, main_loop)
    res = future.result()
    return res
