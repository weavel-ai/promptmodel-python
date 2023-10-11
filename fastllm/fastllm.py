from __future__ import annotations
import asyncio
from dataclasses import dataclass
import dis
import json
import inspect
from typing import Callable, Dict, Any, List, Optional
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import fastllm.utils.logger as logger
from fastllm.llms.llm_proxy import LLMProxy
from fastllm.utils.prompt_util import fetch_prompts

@dataclass
class LLMModule:
    name: str
    default_model: str

class FastLLM:
    """FastLLM main class"""

    def __init__(
        self, default_model: Optional[str] = "gpt-3.5-turbo"
    ):
        self._default_model: str = default_model
        self.llm_modules: List[LLMModule] = []
        self.samples: List[Dict[str, Any]] = []


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
                instruction.opname in ["LOAD_ATTR", "LOAD_METHOD"]
                and instruction.argval == "fastmodel"
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

    def include(self, client: FastLLM):
        self.llm_modules.extend(client.llm_modules)
    
    def get_prompts(self, name: str) -> List[Dict[str, str]]:
        # add name to the list of llm_modules
        self.llm_modules.append(
            LLMModule(
                name=name,
                default_model=self._default_model,
            )
        )
        
        prompts, _ = asyncio.run(fetch_prompts(name))
        return prompts

    def sample(self, name: str, content: Dict[str, Any]):
        self.samples.append(
            {
                "name": name,
                "contents": content
            }
        )

class FastLLMDev(FastLLM):
    def __init__(
        self, default_model: Optional[str] = "gpt-3.5-turbo"
    ):
        super().__init__(default_model)
    
    def include(self, client: FastLLM):
        self.llm_modules.extend(client.llm_modules)