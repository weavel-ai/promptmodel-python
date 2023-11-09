from __future__ import annotations
import asyncio
import dis
import json
import inspect
import threading
import time
import atexit
import nest_asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Union
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import promptmodel.utils.logger as logger
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.prompt_util import update_deployed_db
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils.types import FunctionSchema
from promptmodel.database.orm import initialize_db


@dataclass
class PromptModelInterface:
    name: str


class DevClient:
    """DevClient main class"""

    def __init__(self):
        # if not DevClient._nest_asyncio_applied:
        #     nest_asyncio.apply()
        #     DevClient._nest_asyncio_applied = True
        self.prompt_models: List[PromptModelInterface] = []
        # config = read_config()
        # if (
        #     config
        #     and "dev_branch" in config
        #     and (
        #         (
        #             "online" in config["dev_branch"]
        #             and config["dev_branch"]["online"] == True
        #         )
        #         or (
        #             "initializing" in config["dev_branch"]
        #             and config["dev_branch"]["initializing"] == True
        #         )
        #     )
        # ):
        #     self.cache_manager = None
        # else:
        #     if use_cache:
        #         upsert_config({"use_cache": True}, section="project")
        #         self.cache_manager = CacheManager()
        #     else:
        #         upsert_config({"use_cache": False}, section="project")
        #         self.cache_manager = None
        #         initialize_db()  # init db for local usage

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
                    self.prompt_models.append(
                        PromptModelInterface(name=next_instruction.argval)
                    )

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def register_prompt_model(self, name):
        for prompt_model in self.prompt_models:
            if prompt_model.name == name:
                return

        self.prompt_models.append(PromptModelInterface(name=name))

    def _get_prompt_model_name_list(self) -> List[str]:
        return [prompt_model.name for prompt_model in self.prompt_models]


class DevApp:
    def __init__(self):
        self.prompt_models: List[PromptModelInterface] = []
        self.samples: List[Dict[str, Any]] = []
        self.functions: Dict[str, Dict[str, Union[FunctionSchema, Callable]]] = {}

    def include_client(self, client: DevClient):
        self.prompt_models.extend(client.prompt_models)

    def register_function(
        self, schema: Union[Dict[str, Any], FunctionSchema], function: Callable
    ):
        function_name = schema["name"]
        if isinstance(schema, dict):
            try:
                schema = FunctionSchema(**schema)
            except:
                raise ValueError("schema is not a valid function call schema.")

        if function_name not in self.functions:
            self.functions[function_name] = {
                "schema": schema,
                "function": function,
            }

    def _call_register_function(self, name: str, arguments: Dict[str, str]):
        function_to_call: Callable = self.functions[name]["function"]
        try:
            function_response = function_to_call(**arguments)
            return function_response
        except Exception as e:
            raise e

    def _get_function_name_list(self) -> List[str]:
        return list(self.functions.keys())

    def _get_function_schemas(self, function_names: List[str] = []):
        try:
            function_schemas = [
                self.functions[function_name]["schema"].model_dump()
                for function_name in function_names
            ]
            return function_schemas
        except Exception as e:
            raise e

    def register_sample(self, name: str, content: Dict[str, Any]):
        self.samples.append({"name": name, "contents": content})

    def _get_prompt_model_name_list(self) -> List[str]:
        return [prompt_model.name for prompt_model in self.prompt_models]


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
        asyncio.run(self.update_cache())  # updae cache first synchronously
        self.cache_thread = threading.Thread(target=self._run_cache_loop)
        self.cache_thread.daemon = True
        self.cache_thread.start()
        logger.debug("CacheManager initialized")

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
