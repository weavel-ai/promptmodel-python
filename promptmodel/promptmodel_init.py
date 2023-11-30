import os
import nest_asyncio
import threading
import asyncio
import atexit
import time

from typing import Optional
from datetime import datetime

from promptmodel.utils.config_utils import upsert_config, read_config
from promptmodel.utils import logger
from promptmodel.database.orm import initialize_db
from promptmodel.database.crud import update_deployed_cache
from promptmodel.apis.base import AsyncAPIClient


def init(use_cache: Optional[bool] = True):
    nest_asyncio.apply()

    config = read_config()
    if (
        config
        and "connection" in config
        and (
            (
                "online" in config["connection"]
                and config["connection"]["online"] == True
            )
            or (
                "initializing" in config["connection"]
                and config["connection"]["initializing"] == True
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
                instance = super(CacheManager, cls).__new__(cls)
                instance.last_update_time = 0  # to manage update frequency
                instance.update_interval = 10  # seconds, 6 hours
                instance.program_alive = True
                instance.background_tasks = []
                initialize_db()
                atexit.register(instance._terminate)
                asyncio.run(instance.update_cache())  # updae cache first synchronously
                instance.cache_thread = threading.Thread(
                    target=instance._run_cache_loop
                )
                instance.cache_thread.daemon = True
                instance.cache_thread.start()
                cls._instance = instance
        return cls._instance

    def cache_update_background_task(self, config):
        asyncio.run(update_deployed_db(config))

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
            upsert_config({"version": 0}, section="project")
            config = {"project": {"version": 0}}
        if "project" not in config:
            upsert_config({"version": 0}, section="project")
            config = {"project": {"version": 0}}

        if "version" not in config["project"]:
            upsert_config({"version": 0}, section="project")
            config = {"project": {"version": 0}}

        # Check if we need to update the cache
        if current_time - self.last_update_time > self.update_interval:
            # Update cache logic
            await update_deployed_db(config)
            # Update the last update time
            self.last_update_time = current_time

    def _terminate(self):
        self.program_alive = False

    # async def cleanup_background_tasks(self):
    #     for task in self.background_tasks:
    #         if not task.done():
    #             task.cancel()
    #         try:
    #             await task
    #         except asyncio.CancelledError:
    #             pass  # 작업이 취소됨


async def update_deployed_db(config):
    if "project" not in config or "version" not in config["project"]:
        cached_project_version = 0
    else:
        cached_project_version = int(config["project"]["version"])
    try:
        res = await AsyncAPIClient.execute(
            method="GET",
            path="/check_update",
            params={"cached_version": cached_project_version},
            use_cli_key=False,
        )
        res = res.json()
        if res["need_update"]:
            # update local DB with res['project_status']
            project_status = res["project_status"]
            await update_deployed_cache(project_status)
            upsert_config({"version": res["version"]}, section="project")
        else:
            upsert_config({"version": res["version"]}, section="project")
    except Exception as exception:
        logger.error(f"Deployment cache update error: {exception}")
