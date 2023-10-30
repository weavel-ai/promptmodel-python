from __future__ import annotations
import asyncio
import dis
import json
import inspect
import threading
import time
import atexit
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import promptmodel.utils.logger as logger
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.prompt_util import update_deployed_db
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.database.orm import initialize_db


@dataclass
class LLMModule:
    name: str
    default_model: str = "gpt-3.5-turbo"


class Client:
    """Client main class"""

    def __init__(self, use_cache: Optional[bool] = True, default_model: Optional[str] = "gpt-3.5-turbo"):
        self._default_model: str = default_model
        self.llm_modules: List[LLMModule] = []
        self.samples: List[Dict[str, Any]] = []
        config = read_config()
        if (
            config and 
            "dev_branch" in config and (
                (
                    "online" in config['dev_branch'] and config['dev_branch']["online"] == True
                ) or 
                (
                    "initializing" in config['dev_branch'] and config['dev_branch']["initializing"] == True
                )
            )
        ):
            self.cache_manager = None
        else:
            if use_cache:
                upsert_config({"use_cache": True}, section="project")
                self.cache_manager = CacheManager()
            else:
                upsert_config({"use_cache": False}, section="project")
                self.cache_manager = None
                initialize_db() # init db for local usage

    def fastmodel(self, name: str) -> LLMProxy:
        return LLMProxy(name)

    def register(self, func):
        instructions = list(dis.get_instructions(func))
        for idx in range(
            len(instructions) - 1
        ):  # We check up to len-1 because we access idx+1 inside loop
            instruction = instructions[idx]
            # print(instruction)
            if (
                instruction.opname in ["LOAD_ATTR", "LOAD_METHOD", "LOAD_GLOBAL"]
                and instruction.argval == "PromptModel"
            ):
                next_instruction = instructions[idx + 1]

                # Check if the next instruction is LOAD_CONST with string value
                if next_instruction.opname == "LOAD_CONST" and isinstance(
                    next_instruction.argval, str
                ):
                    self.llm_modules.append(
                        LLMModule(
                            name=next_instruction.argval,
                            default_model=self._default_model,
                        )
                    )

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def register_llm_module(self, name):
        for llm_module in self.llm_modules:
            if llm_module.name == name:
                return

        self.llm_modules.append(
            LLMModule(
                name=name,
                default_model=self._default_model,
            )
        )

    def include(self, client: Client):
        self.llm_modules.extend(client.llm_modules)
        # delete duplicated llm_modules
        self.llm_modules = list(
            {llm_module.name: llm_module for llm_module in self.llm_modules}.values()
        )

        self.samples.extend(client.samples)
        # delete duplicated samples
        self.samples = list(
            {sample["name"]: sample for sample in self.samples}.values()
        )

    def register_sample(self, name: str, content: Dict[str, Any]):
        self.samples.append({"name": name, "contents": content})
        
    def _get_llm_module_name_list(self) -> List[str]:
        return [llm_module.name for llm_module in self.llm_modules]


class DevApp(Client):
    def __init__(self, default_model: Optional[str] = "gpt-3.5-turbo"):
        super().__init__(default_model)

    def include(self, client: Client):
        self.llm_modules.extend(client.llm_modules)


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
        self.update_interval = 5  # seconds, consider tuning this value
        self.program_alive = True
        initialize_db()
        atexit.register(self._terminate)
        # # logger.debug("CacheManager initialized")
        asyncio.run(self.update_cache()) # updae cache first synchronously
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
            config = {"project" : {"version": "0.0.0"}}

        # Check if we need to update the cache
        if current_time - self.last_update_time > self.update_interval:
            # Update cache logic
            await update_deployed_db(config)
            # Update the last update time
            self.last_update_time = current_time

    def _terminate(self):
        self.program_alive = False
