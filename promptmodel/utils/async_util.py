import os
import sys
import yaml
import asyncio
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Tuple, List, Union, Optional, Coroutine
from litellm import token_counter

from promptmodel.apis.base import AsyncAPIClient
from promptmodel.database.crud import (
    get_latest_version_prompts,
    get_deployed_prompts,
)
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils import logger
from promptmodel.utils.random_utils import select_version_by_ratio
from promptmodel.promptmodel_init import CacheManager, update_deployed_db


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
