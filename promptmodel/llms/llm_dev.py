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
        model: Optional[str] = None,
        output_keys: List[str] = [],
    ) -> AsyncGenerator[Dict[str, str], None]:
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
        )
        async for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                stream_value = chunk["choices"][0]["delta"]["content"]
                raw_output += stream_value  # 지금까지 생성된 누적 output
                yield stream_value  # return raw output 
        
        # parsing
        if parsing_type:
            parsing_pattern : Dict[str, str] = get_pattern_by_type(parsing_type)
            whole_pattern = parsing_pattern['whole']
            parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
            cannot_parsed_output = re.sub(whole_pattern, "", raw_output, flags=re.DOTALL)
            if cannot_parsed_output.strip() != "":
                yield False
            else:
                yield True
            for parsed_result in parsed_results:
                yield {parsed_result[0] : parsed_result[1]}
        
        # if not parsing_type:
        #     async for chunk in response:
        #        if "content" in chunk["choices"][0]["delta"]:
        #             stream_value = chunk["choices"][0]["delta"]["content"]
        #             yield stream_value  # return raw output 
        # else:
        #     if parsing_type == ParsingType.COLON.value:
        #         # TODO: implement colon parsing
        #         pass
        #     else:
        #         parsing_pattern : Dict[str, str] = get_pattern_by_type(parsing_type)
        #         expected_pattern_keys = "|".join(map(re.escape, output_keys))
        #         whole_pattern: str = parsing_pattern["whole"].format(key=expected_pattern_keys)
        #         start_pattern: str = parsing_pattern["start"].format(key=expected_pattern_keys)
        #         end_pattern_raw: str = parsing_pattern["end"]
        #         end_flag: str = parsing_pattern["end_flag"]
        #         refresh_flag: str = parsing_pattern["refresh_flag"]
        #         if parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
        #             terminate_flag = "]]"
        #             end_identifier = "[["
                    
        #             pause_stream = False
        #             pause_cache = ""
        #             streaming_key = None
        #             async for chunk in response:
        #                 if "content" in chunk["choices"][0]["delta"]:
        #                     stream_value = chunk["choices"][0]["delta"]["content"]
        #                     yield stream_value  # return raw output

        #                     raw_output += stream_value  # 지금까지 생성된 누적 output

        #                     stripped_output = re.sub(
        #                         whole_pattern, "", raw_output, flags=re.DOTALL
        #                     )  # 누적 output에서 [key start] ~ [key end] 부분을 제거한 output
                            
        #                     new_streaming_key = re.findall(
        #                         start_pattern,
        #                         stripped_output,
        #                         flags=re.DOTALL,  # Find start pattern in stripped output
        #                     )

        #                     if len(streaming_key) > 1:
        #                         raise ValueError("Multiple Matches")
                            
        #                     if not streaming_key:
        #                         if not new_streaming_key: # If no start pattern found, continue
        #                             continue
        #                         streaming_key = new_streaming_key[0]
                            
        #                     end_pattern = end_pattern_raw.format(key=streaming_key) # end pattern should have same key with start pattern
                            
        #                     if stream_value.find(end_flag) != -1: # if 1st token of end pattern apears in stream value, pause stream
        #                         pause_stream = True
                            
        #                     if pause_stream: # if pause stream, cache stream value
        #                         pause_cache += stream_value
        #                     else:
        #                         yield {streaming_key : stream_value}
                                
        #                     if stream_value.find(refresh_flag) != -1: # if last token of end pattern apears in stream value,
        #                         if pause_cache.find(end_identifier) == -1: # if "[[" is not in pause cache
        #                             if stream_value.find(end_flag) != -1: # if stream_value is like "]["
        #                                 # yield only before end_flag & save after end_flag to pause cache (save with end_flag)
        #                                 yield {streaming_key : pause_cache.split(end_flag)[0].replace(end_flag, "")}
        #                                 pause_cache = end_flag + pause_cache.split(end_flag)[1]
        #                                 pause_stream = True
        #                             else:
        #                                 yield {streaming_key : pause_cache}
        #                                 pause_cache = ""
        #                                 pause_stream = False
        #                         elif terminate_flag in pause_cache: # if ]] is in pause cache
        #                             if pause_cache.find(end_pattern) != -1: # if there is end pattern
        #                                 yield {streaming_key : pause_cache.split(end_pattern)[0].replace(end_pattern, "")}
        #                                 streaming_key = None
        #                                 pause_cache = ""
        #                                 pause_stream = False
        #                             else:
        #                                 raise ValueError(f"Parsing Error, \"{pause_cache}\" is forbidden.")
        #                         else: # generating psuedo end pattern
        #                             continue
                                                            
        #             # if streaming end but there is still pause stream, parsing error
        #             if streaming_key:
        #                 raise ValueError("Parsing Error. End pattern did not be generated properly.")
                    
        #         else:
                    
        #             pause_stream = False
        #             pause_cache = ""
        #             streaming_key = None
        #             async for chunk in response:
        #                 if "content" in chunk["choices"][0]["delta"]:
        #                     stream_value = chunk["choices"][0]["delta"]["content"]
        #                     yield stream_value  # return raw output

        #                     raw_output += stream_value  # 지금까지 생성된 누적 output

        #                     stripped_output = re.sub(
        #                         whole_pattern, "", raw_output, flags=re.DOTALL
        #                     )  # 누적 output에서 [key start] ~ [key end] 부분을 제거한 output
                            
        #                     new_streaming_key = re.findall(
        #                         start_pattern,
        #                         stripped_output,
        #                         flags=re.DOTALL,  # Find start pattern in stripped output
        #                     )

        #                     if len(streaming_key) > 1:
        #                         raise ValueError("Multiple Matches")
                            
        #                     if not streaming_key:
        #                         if not new_streaming_key: # If no start pattern found, continue
        #                             continue
        #                         streaming_key = new_streaming_key[0]
                            
        #                     end_pattern = end_pattern_raw.format(key=streaming_key) # end pattern should have same key with start pattern
                            
        #                     if stream_value.find(end_flag) != -1: # if 1st token of end pattern apears in stream value, pause stream
        #                         pause_stream = True
                            
        #                     if pause_stream: # if pause stream, cache stream value
        #                         pause_cache += stream_value
        #                     else:
        #                         yield {streaming_key : stream_value}
                                
        #                     if stream_value.find(refresh_flag) != -1: # if last token of end pattern apears in stream value,
        #                         if pause_cache.find(end_pattern) != -1: # if there is end pattern
        #                             yield {streaming_key : pause_cache.split(end_pattern)[0].replace(end_pattern, "")}
        #                             streaming_key = None
        #                             pause_cache = ""
        #                             pause_stream = False
        #                         else:
        #                             raise ValueError(f"Parsing Error, \"{pause_cache}\" is forbidden.")
                                                            
        #             # if streaming end but there is still pause stream, parsing error
        #             if streaming_key:
        #                 raise ValueError("Parsing Error. End pattern did not be generated properly.")
                    
