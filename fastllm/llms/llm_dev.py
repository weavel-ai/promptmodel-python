"""LLM module for Development TestRun"""
import re
import os
import json
import openai

from typing import Any, AsyncGenerator, List, Dict, Optional, Union, Generator
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import completion

import fastllm.utils.logger as logger

load_dotenv()

class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class ParsingType:
    COLON = "colon"
    SQURE_BRACKET = "square_bracket"
    DOUBLE_SQURE_BRACKET = "double_square_bracket"

class OpenAIMessage(BaseModel):
    role: str
    content: str
    
class LLMDev:
    def __init__(self):
        self._model: str
        
    async def dev_generate(
        self,
        messages: List[Dict[str, str]],
        parsing_type: ParsingType,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        response = await openai.ChatCompletion.acreate(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )
        
        if parsing_type == ParsingType.DOUBLE_SQURE_BRACKET:
            async for chunk in response:
                pause_stream = False
                if "content" in chunk["choices"][0]["delta"]:
                    stream_value = chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value  # 지금까지 생성된 누적 output
                    pattern = (
                        r"\[\[.*?(\s*\(.+\))?\sstart\]\](.*?)\[\[.*?(\s*\(.+\))?\send\]\]"
                    )
                    stripped_output = re.sub(
                        pattern, "", raw_output, flags=re.DOTALL
                    )  # 누적 output에서 [key start] ~ [key end] 부분을 제거한 output
                    streaming_key = re.findall(
                        r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                        stripped_output,
                        flags=re.DOTALL,  # stripped output에서 [key start] 부분을 찾음
                    )
                    if not streaming_key:  # 아직 output value를 streaming 중이 아님
                        continue

                    if len(streaming_key) > 1:
                        raise ValueError("Multiple Matches")
                    # key = streaming_key[0].lower()
                    key = streaming_key[0]
                    
                    if stream_value.find("]") != -1 or "[" in re.sub(
                        r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                        "",
                        stripped_output.split(f"[[{key} start]]")[-1],
                        flags=re.DOTALL,
                    ):  # 현재 stream 중인 output이 [[key end]] 부분일 경우에는 pause_stream을 True로 설정
                        if stream_value.find("[") != -1:
                            if cache.find("[[") != -1:
                                logger.info("[[ in cache")
                                pause_stream = True
                            else:
                                cache += "["
                        pause_stream = True
                    if not pause_stream:
                        yield stream_value  # return raw output
                        yield {key: stream_value} # return parsed output
                    elif stream_value.find("]") != -1:
                        # Current stream_value (that includes ]) isn't yielded, but the next stream_values will be yielded.
                        cache = ""
                        pause_stream = False
        else:
            # TODO: add other parsing types
            pass