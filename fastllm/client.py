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

import fastllm.utils.logger as logger
from fastllm.llms.llm_proxy import LLMProxy
from fastllm.utils.prompt_util import fetch_prompts, update_deployed_db
from fastllm.utils.config_utils import read_config, upsert_config
from fastllm.database.orm import initialize_db

@dataclass
class LLMModule:
    name: str
    default_model: str = "gpt-3.5-turbo"

class Client:
    """PromptModel Client class (Singleton)"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Client, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.llm_modules: List[LLMModule] = []
        self.samples: List[Dict[str, Any]] = []
        config = read_config()
        dev_branch = config["dev_branch"]
        if ("online" in dev_branch and dev_branch["online"] == True) or ("initializing" in dev_branch and dev_branch["initializing"] == True):
            self.cache_manager = None
        else:
            self.cache_manager = CacheManager() # only in deployment mode
        # logger.debug("FastLLM initialized")

    def register(self, func):
        instructions = list(dis.get_instructions(func))
        for idx in range(
            len(instructions) - 1
        ):  # We check up to len-1 because we access idx+1 inside loop
            instruction = instructions[idx]
            # print(instruction)
            if (
                instruction.opname in ["LOAD_ATTR", "LOAD_METHOD"]
                and instruction.argval == "PromptModel"
            ):
                next_instruction = instructions[idx + 1]

                # Check if the next instruction is LOAD_CONST with string value
                if next_instruction.opname == "LOAD_CONST" and isinstance(
                    next_instruction.argval, str
                ):
                    self.llm_modules.append(
                        LLMModule(
                            name=next_instruction.argval
                        )
                    )

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def sample(self, name: str, content: Dict[str, Any]):
        self.samples.append(
            {
                "name": name,
                "contents": content
            }
        )

class DevApp(Client):
    def __init__(
        self
    ):
        super().__init__()
    

class CacheManager:
    """CacheManager class to manage the cache of the deployed PromptModels"""    
    
    def __init__(self):
        self.last_update_time = 0 # to manage update frequency
        self.update_interval = 5 # seconds, consider tuning this value
        self.program_alive = True
        initialize_db()
        atexit.register(self._terminate)
        logger.debug("CacheManager initialized")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._update_cache_periodically())
        loop.close()
    
    async def _update_cache_periodically(self):
        while True:
            await self.update_cache()
            # logger.debug("Update cache")
            await asyncio.sleep(self.update_interval)  # Non-blocking sleep
    
    async def update_cache(self):
        # Current time
        current_time = time.time()
        config = read_config()
        
        # Check if we need to update the cache
        if current_time - self.last_update_time > self.update_interval:
            # Update cache logic
            await update_deployed_db(config)
            # Update the last update time
            self.last_update_time = current_time

    def _terminate(self):
        self.program_alive = False