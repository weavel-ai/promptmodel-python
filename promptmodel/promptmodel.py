from __future__ import annotations
import asyncio
import dis
import json
import inspect
import threading
import time
import atexit
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from litellm import RateLimitManager

import promptmodel.utils.logger as logger
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.prompt_util import fetch_prompts, update_deployed_db
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.database.orm import initialize_db
from promptmodel import Client


@dataclass
class LLMModule:
    name: str
    default_model: str


class RegisteringMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        # Find the global client instance in the current module
        client = cls.find_client_instance()
        if client is not None:
            client.register_llm_module(instance.name)
        return instance

    @staticmethod
    def find_client_instance():
        import sys

        # Get the current frame
        frame = sys._getframe(2)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of Client among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, Client):
                return var_val
        return None


class PromptModel(metaclass=RegisteringMeta):
    def __init__(self, name, rate_limit_manager: Optional[RateLimitManager] = None):
        self.name = name
        self.llm_proxy = LLMProxy(name, rate_limit_manager)

    def get_prompts(self) -> List[Dict[str, str]]:
        # add name to the list of llm_modules

        prompts, _, _ = asyncio.run(fetch_prompts(self.name))
        return prompts

    def run(self, inputs: Dict[str, Any] = {}) -> str:
        return self.llm_proxy.run(inputs)

    async def arun(self, inputs: Dict[str, Any] = {}) -> str:
        return await self.arun(inputs)

    def stream(self, inputs: Dict[str, Any] = {}) -> Generator[str, None, None]:
        for item in self.llm_proxy.stream(inputs):
            yield item

    async def astream(
        self, inputs: Optional[Dict[str, Any]] = {}
    ) -> AsyncGenerator[str, None]:
        async for item in self.llm_proxy.astream(inputs):
            yield item

    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Dict[str, str]:
        return self.llm_proxy.run_and_parse(inputs, output_keys)

    async def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Dict[str, str]:
        return await self.llm_proxy.arun_and_parse(inputs, output_keys)

    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Generator[str, None, None]:
        for item in self.llm_proxy.stream_and_parse(inputs, output_keys):
            yield item

    async def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> AsyncGenerator[str, None]:
        async for item in self.llm_proxy.astream_and_parse(inputs, output_keys):
            yield item

    def run_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> Generator[str, None, None]:
        return self.llm_proxy.run_and_parse_function_call(inputs, function_list)

    async def arun_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> Generator[str, None, None]:
        return await self.llm_proxy.arun_and_parse_function_call(inputs, function_list)

    async def astream_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> AsyncGenerator[str, None]:
        async for item in self.llm_proxy.astream_and_parse_function_call(
            inputs, function_list
        ):
            yield item
