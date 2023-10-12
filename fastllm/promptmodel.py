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
from fastllm.client import Client

@dataclass
class LLMModule:
    name: str
    default_model: str

# class RegisteringMeta(type):
#     def __call__(cls, *args, **kwargs):
#         instance = super().__call__(*args, **kwargs)
#         # Use the global client instance
#         client.register_instance(instance)
#         return instance 
