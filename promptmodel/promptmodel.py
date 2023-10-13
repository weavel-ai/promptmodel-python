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

from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.prompt_util import fetch_prompts
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
            client.register_instance(instance)
        return instance 

    @staticmethod
    def find_client_instance():
        import sys
        # Get the current frame
        frame = sys._getframe(1)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of Client among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, Client):
                return var_val
        return None
    
class PromptModel(metaclass=RegisteringMeta):
    def __init__(self, name):
        self.name = name
        self.proxy = LLMProxy(name)
        
    # def __del__(self):
    #     try:
    #         client = Client.get_global_instance()
    #         client.unregister_instance(self)
    #     except Exception as e:
    #         pass
        
    def prompts(self):
        return asyncio.run(fetch_prompts(self.name))
    
    
    def generate(self, inputs: Dict[str, Any] = {}) -> str:
        return self.proxy.generate(inputs)

    def stream(self, inputs: Dict[str, Any] = {}) -> Generator[str, None, None]:
        for item in self.proxy.stream(inputs):
            yield item

    def generate_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Dict[str, str]:
        return self.proxy.generate_and_parse(inputs, output_keys)

    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Generator[str, None, None]:
        for item in self.proxy.stream_and_parse(inputs, output_keys):
            yield item

    def generate_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> Dict[str, Any]:
        return self.proxy.generate_and_parse_function_call(inputs, function_list)

    async def agenerate(self, inputs: Dict[str, Any] = {}) -> str:
        return await self.proxy.agenerate(inputs)

    async def astream(self, inputs: Optional[Dict[str, Any]] = {}) -> AsyncGenerator[str, None]:
        async for item in self.proxy.astream(inputs):
            yield item

    async def agenerate_and_parse(self, inputs: Dict[str, Any] = {}, output_keys: List[str] = []) -> Dict[str, str]:
        return await self.proxy.agenerate_and_parse(inputs, output_keys)

    async def astream_and_parse(self, inputs: Dict[str, Any] = {}, output_keys: List[str] = []) -> AsyncGenerator[str, None]:
        async for item in self.proxy.astream_and_parse(inputs, output_keys):
            yield item

    async def agenerate_and_parse_function_call(self, inputs: Dict[str, Any] = {}, function_list: List[Callable[..., Any]] = []) -> Dict[str, Any]:
        return await self.proxy.agenerate_and_parse_function_call(inputs, function_list)

    async def astream_and_parse_function_call(self, inputs: Dict[str, Any] = {}, function_list: List[Callable[..., Any]] = []) -> AsyncGenerator[str, None]:
        async for item in self.proxy.astream_and_parse_function_call(inputs, function_list):
            yield item