from __future__ import annotations
import dis
import nest_asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Union

from promptmodel.types.response import FunctionSchema


@dataclass
class FunctionModelInterface:
    name: str


@dataclass
class ChatModelInterface:
    name: str


class DevClient:
    """DevClient main class"""

    def __init__(self):
        self.function_models: List[FunctionModelInterface] = []
        self.chat_models: List[ChatModelInterface] = []

    def register(self, func):
        instructions = list(dis.get_instructions(func))
        for idx in range(
            len(instructions) - 1
        ):  # We check up to len-1 because we access idx+1 inside loop
            instruction = instructions[idx]
            # print(instruction)
            if instruction.opname in ["LOAD_ATTR", "LOAD_METHOD", "LOAD_GLOBAL"] and (
                instruction.argval == "FunctionModel"
                or instruction.argval == "ChatModel"
            ):
                next_instruction = instructions[idx + 1]

                # Check if the next instruction is LOAD_CONST with string value
                if next_instruction.opname == "LOAD_CONST" and isinstance(
                    next_instruction.argval, str
                ):
                    if instruction.argval == "FunctionModel":
                        self.function_models.append(
                            FunctionModelInterface(name=next_instruction.argval)
                        )
                    elif instruction.argval == "ChatModel":
                        self.chat_models.append(
                            ChatModelInterface(name=next_instruction.argval)
                        )

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def register_function_model(self, name):
        for function_model in self.function_models:
            if function_model.name == name:
                return

        self.function_models.append(FunctionModelInterface(name=name))

    def register_chat_model(self, name):
        for chat_model in self.chat_models:
            if chat_model.name == name:
                return

        self.chat_models.append(ChatModelInterface(name=name))

    def _get_function_model_name_list(self) -> List[str]:
        return [function_model.name for function_model in self.function_models]


class DevApp:
    _nest_asyncio_applied = False

    def __init__(self):
        self.function_models: List[FunctionModelInterface] = []
        self.chat_models: List[ChatModelInterface] = []
        self.samples: List[Dict[str, Any]] = []
        self.functions: Dict[
            str, Dict[str, Union[FunctionSchema, Optional[Callable]]]
        ] = {}

        if not DevApp._nest_asyncio_applied:
            DevApp._nest_asyncio_applied = True
            nest_asyncio.apply()

    def include_client(self, client: DevClient):
        self.function_models.extend(client.function_models)
        self.chat_models.extend(client.chat_models)

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
        function_to_call: Optional[Callable] = self.functions[name]["function"]
        if not function_to_call:
            return
        try:
            function_response = function_to_call(**arguments)
            return function_response
        except Exception as e:
            raise e

    def _get_function_name_list(self) -> List[str]:
        return list(self.functions.keys())

    def _get_function_schema_list(self) -> List[Dict]:
        return [
            self.functions[function_name]["schema"].model_dump()
            for function_name in self._get_function_name_list()
        ]

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
        self.samples.append({"name": name, "content": content})

    def _get_function_model_name_list(self) -> List[str]:
        return [function_model.name for function_model in self.function_models]

    def _get_chat_model_name_list(self) -> List[str]:
        return [chat_model.name for chat_model in self.chat_models]
