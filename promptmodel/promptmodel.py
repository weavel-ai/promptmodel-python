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
from promptmodel.utils.prompt_util import fetch_prompts, update_deployed_db
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.database.orm import initialize_db
from promptmodel import Client

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
