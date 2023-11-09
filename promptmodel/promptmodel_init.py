import os
import nest_asyncio
import threading
import asyncio
import atexit
import time

from typing import Optional

from promptmodel.utils.config_utils import upsert_config, read_config
from promptmodel.database.orm import initialize_db
from promptmodel.utils.prompt_util import update_deployed_db


def init(use_cache: Optional[bool] = True):
    nest_asyncio.apply()

    config = read_config()
    if (
        config
        and "dev_branch" in config
        and (
            (
                "online" in config["dev_branch"]
                and config["dev_branch"]["online"] == True
            )
            or (
                "initializing" in config["dev_branch"]
                and config["dev_branch"]["initializing"] == True
            )
        )
    ):
        cache_manager = None
    else:
        if use_cache:
            upsert_config({"use_cache": True}, section="project")
            cache_manager = CacheManager()
        else:
            upsert_config({"use_cache": False}, section="project")
            cache_manager = None
            initialize_db()  # init db for local usage


class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.last_update_time = 0  # to manage update frequency
        self.update_interval = 60 * 60 * 6  # seconds, 6 hours
        self.program_alive = True
        initialize_db()
        atexit.register(self._terminate)
        asyncio.run(self.update_cache())  # updae cache first synchronously
        self.cache_thread = threading.Thread(target=self._run_cache_loop)
        self.cache_thread.daemon = True
        self.cache_thread.start()

    def _run_cache_loop(self):
        asyncio.run(self._update_cache_periodically())

    async def _update_cache_periodically(self):
        while True:
            await asyncio.sleep(self.update_interval)  # Non-blocking sleep
            await self.update_cache()

    async def update_cache(self):
        # Current time
        current_time = time.time()
        config = read_config()
        if not config:
            upsert_config({"version": "0.0.0"}, section="project")
            config = {"project": {"version": "0.0.0"}}

        # Check if we need to update the cache
        if current_time - self.last_update_time > self.update_interval:
            # Update cache logic
            await update_deployed_db(config)
            # Update the last update time
            self.last_update_time = current_time

    def _terminate(self):
        self.program_alive = False
