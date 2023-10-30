"""LLM module for Development TestRun"""
import re
import os
import json
import openai

from typing import Any, AsyncGenerator, List, Dict, Optional, Union, Generator
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import acompletion

from promptmodel.utils.enums import ParsingType, ParsingPattern, get_pattern_by_type
from promptmodel.utils import logger
from promptmodel.utils.output_utils import convert_str_to_type
from promptmodel.utils.types import LLMStreamResponse

load_dotenv()


class OpenAIMessage(BaseModel):
    role: str
    content: str


class LLMDev:
    def __init__(self):
        self._model: str

    def __validate_openai_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        return [OpenAIMessage(**message) for message in messages]

    async def dev_run(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: [List[Any]] = [],
        model: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        response = await acompletion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
            functions=functions,
        )
        function_call = {"name" : "", "arguments" : ""}
        async for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                stream_value = chunk["choices"][0]["delta"]["content"]
                raw_output += stream_value # append raw output
                yield LLMStreamResponse(raw_output=stream_value) # return raw output
                
            if "function_call" in chunk["choices"][0]["delta"]:
                for key, value in chunk["choices"][0]["delta"]["function_call"].items():
                    function_call[key] = value

            if chunk["choices"][0]["finish_reason"] == "function_call":
                yield LLMStreamResponse(function_call=function_call)
                
        # parsing
        if parsing_type:
            parsing_pattern: Dict[str, str] = get_pattern_by_type(parsing_type)
            whole_pattern = parsing_pattern["whole"]
            parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
            for parsed_result in parsed_results:
                key = parsed_result[0]
                type_str = parsed_result[1]
                value = convert_str_to_type(parsed_result[2], type_str)
                yield LLMStreamResponse(parsed_outputs={key: value})
