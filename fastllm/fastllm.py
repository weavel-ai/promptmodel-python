import asyncio
from dataclasses import dataclass
import dis
import json
import inspect
from os import name
from typing import Callable, Dict, Any, List, Optional
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from .llms.llm_proxy import LLMProxy

from .llms.llm import LLM

from .utils import logger
from fastllm.fastllm import FastLLM

GATEWAY_URL = "wss://agentlabs.up.railway.app/ws/agent"


@dataclass
class LLMModule:
    name: str
    default_model: str


class FastLLM:
    """FastLLM main class"""

    def __init__(
        self, name: Optional[str], default_model: Optional[str] = "gpt-3.5-turbo"
    ):
        self._name: str = name
        self._default_model: str = default_model
        self.llm_modules: List[LLMModule] = []

    def llm(self, name: str) -> LLMProxy:
        return LLMProxy(name)

    def with_llm(self, func):
        instructions = list(dis.get_instructions(func))
        for idx in range(
            len(instructions) - 1
        ):  # We check up to len-1 because we access idx+1 inside loop
            instruction = instructions[idx]
            print(instruction)
            if (
                instruction.opname in ["LOAD_ATTR", "LOAD_METHOD"]
                and instruction.argval == "llm"
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

    def register(self, client: FastLLM):
        self.llm_modules.extend(client.llm_modules)
