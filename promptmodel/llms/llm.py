"""Base module for interacting with LLM APIs."""
import re
import os
import json
import time
import datetime
from typing import (
    Any, AsyncGenerator, List, Dict, Optional, Union, Generator, Tuple
)

import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import completion, acompletion
from litellm import ModelResponse, RateLimitManager
from litellm.utils import prompt_token_calculator, token_counter

from promptmodel.utils import logger
from promptmodel.utils.enums import ParsingType, ParsingPattern, get_pattern_by_type

load_dotenv()


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class OpenAIMessage(BaseModel):
    role: str
    content: str


DEFAULT_MODEL = "gpt-3.5-turbo"


class LLM:
    def __init__(self, rate_limit_manager: Optional[RateLimitManager] = None):
        self._rate_limit_manager = rate_limit_manager

    @classmethod
    def __parse_output__(cls, raw_output: str, key: str) -> Union[str, None]:
        """Parse value for key from raw output."""
        # capitalized_key = key[0].upper() + key[1:]  # capitalize the first letter of the key
        pattern = r"\[\[{key}(\s*\(.+\))?\sstart\]\](.*?)\[\[{key}(\s*\(.+\))?\send\]\]".format(
            # key=capitalized_key
            key=key
        )
        results = re.findall(pattern, raw_output, flags=re.DOTALL)
        results = [result[1] for result in results]
        if results:
            if len(results) > 1:
                raise ValueError("Multiple Matches")
            return results[0].strip()
        else:
            return None

    @classmethod
    def __parse_outputs__(cls, raw_output: str) -> Dict[str, str]:
        """Parse all key-value pairs from raw output."""
        pattern = (
            r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\](.*?)\[\[\1(?:\s*\(.+\))?\send\]\]"
        )
        matches = re.findall(pattern, raw_output, flags=re.DOTALL)

        parsed_outputs = {}
        for key, value in matches:
            parsed_key = key.strip()
            parsed_value = value.strip()
            if parsed_key in parsed_outputs:
                raise ValueError(f"Multiple Matches for key: {parsed_key}")
            parsed_outputs[parsed_key] = parsed_value

        return parsed_outputs

    @classmethod
    def __parse_output_pattern__(
        cls,
        raw_output: str,
        parsing_type: ParsingType
    ) -> (Dict[str, str], bool):
        parsing_pattern = get_pattern_by_type(parsing_type)
        whole_pattern = parsing_pattern['whole']
        parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
        parsed_outputs = {}
        
        for parsed_result in parsed_results:
            parsed_outputs[parsed_result[0]] = parsed_result[1]
        
        cannot_parsed_output = re.sub(whole_pattern, "", raw_output, flags=re.DOTALL)
        
        if cannot_parsed_output.strip() != "":
            return parsed_outputs, False
        else:
            return parsed_outputs, True
        
    def __validate_openai_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        return [OpenAIMessage(**message) for message in messages]

    def run(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        )
        res = response.choices[0]["message"]["content"]
        if show_response:
            return res, response
        return res

    def run_function_call(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any],
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            function_call=True,
            functions=functions,
        )
        content = response.choices[0]["message"]["content"]
        call_func = (
            response.choices[0]["message"]["function_call"]
            if "function_call" in response.choices[0]["message"]
            else None
        )
        if show_response:
            return content, call_func, response
        return content, call_func

    async def arun(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        else:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        res = response.choices[0]["message"]["content"]
        if show_response:
            return res, response
        return res

    async def arun_function_call(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any],
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                function_call=True,
                functions=functions,
            )
        else:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                function_call=True,
                functions=functions,
            )
        content = (
            response.choices[0]["message"]["content"]
            if "content" in response.choices[0]["message"]
            else None
        )
        call_func = (
            response.choices[0]["message"]["function_call"]
            if "function_call" in response.choices[0]["message"]
            else None
        )
        if show_response:
            return content, call_func, response
        return content, call_func

    def stream(
        self,
        messages: List[Dict[str, str]],  # input
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ):
        """Stream openai chat completion."""
        # load_prompt()
        start_time = datetime.datetime.now()
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        ).choices[0]["message"]["content"]

        raw_output = ""
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                raw_output += chunk["choices"][0]["delta"]["content"]
                yield chunk["choices"][0]["delta"]["content"]
            if chunk["choices"][0]["finish_reason"] != None:
                if show_response:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(
                        chunk, response_ms, messages, raw_output
                    )

    def run_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: ParsingType,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ) -> Dict[str, str]:
        """Parse and return output from openai chat completion."""
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        )
        raw_output = response.choices[0]["message"]["content"]

        parsed_output, parsed_success = self.__parse_output_pattern__(raw_output, parsing_type)

        if show_response:
            return parsed_output, parsed_success, response
        return parsed_output, parsed_success

    def stream_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        model: Optional[str] = DEFAULT_MODEL,
        **kwargs,
    ) -> Generator[Dict[str, str], None, None]:
        """Parse & stream output from openai chat completion."""
        if parsing_type == ParsingType.COLON.value:
            # cannot stream colon type
            yield False
            return
        start_time = datetime.datetime.now()
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )
        
        if parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
            return self.__double_type_sp_generator__(messages, response, parsing_type, start_time)
        else:
            return self.__single_type_sp_generator__(messages, response, parsing_type, start_time)

    async def arun_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ) -> Dict[str, str]:
        """Generate openai chat completion asynchronously, and parse the output.
        Example prompt is as follows:
        -----
        Given a topic, you are required to generate a story.
        You must follow the provided output format.

        Topic:
        {topic}

        Output format:
        [Story start]
        ...
        [Story end]

        Now generate the output:
        """
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        else:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        raw_output = response.choices[0]["message"]["content"]
        # # logger.debug(f"Output:\n{raw_output}")
        if output_keys is None:
            parsed_output = self.__parse_outputs__(raw_output)
        else:
            parsed_output = {}
            for key in output_keys:
                output = self.__parse_output__(raw_output, key)
                if output:
                    parsed_output[key] = output

        if show_response:
            return parsed_output, response
        return parsed_output

    async def astream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ) -> Generator[Dict[str, str], None, None]:
        """Parse & stream output from openai chat completion."""
        start_time = datetime.datetime.now()
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
            )
        else:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
            )
        raw_output = ""
        async for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                raw_output += chunk["choices"][0]["delta"]["content"]
                yield chunk["choices"][0]["delta"]["content"]
            if chunk["choices"][0]["finish_reason"] != None:
                if show_response:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(
                        chunk, response_ms, messages, raw_output
                    )

    async def astream_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        show_response: bool = False,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Parse & stream output from openai chat completion."""
        if parsing_type == ParsingType.COLON.value:
            # cannot stream colon type
            yield False
            return
        start_time = datetime.datetime.now()
        response = await acompletion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )
        
        if parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
            async for result in self.__double_type_sp_agenerator__(messages, response, parsing_type, start_time):
                yield result
        else:
            async for result in self.__single_type_sp_agenerator__(messages, response, parsing_type, start_time):
                yield result



    async def aget_embedding(self, context: str) -> List[float]:
        """
        Return the embedding of the context.
        """
        context = context.replace("\n", " ")
        response = await openai.Embedding.acreate(
            input=[context], model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    def make_model_response(
        self,
        chunk: dict,
        response_ms,
        messages: List[Dict[str, str]],
        raw_output: str,
        function_call: Optional[dict] = None,
    ) -> ModelResponse:
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": raw_output},
                "finish_reason": chunk["choices"][0]["finish_reason"],
            }
        ]
        if function_call:
            choices[0]["message"]["function_call"] = function_call
        prompt_token: int = prompt_token_calculator(chunk["model"], messages)
        completion_token: int = token_counter(chunk["model"], raw_output)
        usage = {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens": prompt_token + completion_token,
        }
        res = ModelResponse(
            id=chunk["id"],
            choices=choices,
            created=chunk["created"],
            model=chunk["model"],
            usage=usage,
            response_ms=response_ms,
        )
        return res

    def __double_type_sp_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator,
        parsing_type: ParsingType,
        start_time: datetime.datetime
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern['start']
            start_fstring = parsing_pattern['start_fstring']
            end_fstring = parsing_pattern['end_fstring']
            start_token = parsing_pattern['start_token']
            end_token = parsing_pattern['end_token']
            
            end_tag_identifier = "[[/" if parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value else "[[/"
            
            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    stream_value : str= chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value 
                    yield stream_value
                    buffer += stream_value
                    
                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield False
                                break
                            if len(keys) == 0:
                                break # no key
                            active_key = keys[0]
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(key=active_key)
                            buffer = buffer.split(start_pattern)[-1]
                            
                        else:
                            if stream_value.find(start_token) != -1: # start token appers in chunk -> pause
                                stream_pause = True 
                                break
                            elif stream_pause:
                                if buffer.find(end_tag) != -1: # if end tag appears in buffer
                                    yield {active_key: buffer.split(end_tag)[0]}
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                    break
                                elif stream_value.find(end_token) != -1: # if ("[blah]" != end_pattern) appeared in buffer 
                                    if buffer.find(end_token + end_token) != -1: # if ]] in buffer -> error
                                        yield False
                                        yield {active_key: buffer.split(start_token)[0]}
                                        buffer = buffer.split(end_token + end_token)[-1]
                                        stream_pause = False
                                        break
                                    else:
                                        if buffer.find(start_token + start_token) != -1: # if [[ in buffer -> pause
                                            break
                                        else:
                                            # if [ in buffer (== [blah]) -> stream
                                            yield {active_key: buffer}
                                            buffer = ""
                                            stream_pause = False
                                            break
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield {active_key: buffer}
                                    buffer = ""
                                break
                            
                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(
                        chunk, response_ms, messages, raw_output
                    )
        except Exception as e:
            logger.error(e)
            yield False

    def __single_type_sp_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator,
        parsing_type: ParsingType,
        start_time: datetime.datetime
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern['start']
            start_fstring = parsing_pattern['start_fstring']
            end_fstring = parsing_pattern['end_fstring']
            start_token = parsing_pattern['start_token']
            end_token = parsing_pattern['end_token']
            
            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    stream_value : str= chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value 
                    yield stream_value
                    buffer += stream_value
                    
                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield False
                                break
                            if len(keys) == 0:
                                break # no key
                            
                            active_key = keys[0]
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(key=active_key)
                            buffer = buffer.split(start_pattern)[-1]
                            
                        else:
                            if stream_value.find(start_token) != -1: # start token appers in chunk -> pause
                                stream_pause = True 
                                break
                            elif stream_pause:
                                if buffer.find(end_tag) != -1: # if end tag appears in buffer
                                    yield {active_key: buffer.split(end_tag)[0].replace(end_tag, "")}
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                elif stream_value.find(end_token) != -1: # if pattern ends  = ("[blah]" != end_pattern) appeared in buffer 
                                    yield False # multiple matching error
                                    stream_pause = False
                                    yield {active_key: buffer}
                                    buffer = ""
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield {active_key: buffer}
                                    buffer = ""
                                break
                            
                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(
                        chunk, response_ms, messages, raw_output
                    )
        except Exception as e:
            logger.error(e)
            yield False
            
    async def __double_type_sp_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator,
        parsing_type: ParsingType,
        start_time: datetime.datetime
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern['start']
            start_fstring = parsing_pattern['start_fstring']
            end_fstring = parsing_pattern['end_fstring']
            start_token = parsing_pattern['start_token']
            end_token = parsing_pattern['end_token']
            
            end_tag_identifier = "[[/" if parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value else "[[/"
            
            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            async for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    stream_value : str= chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value 
                    yield stream_value
                    buffer += stream_value
                    
                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield False
                                break
                            if len(keys) == 0:
                                break # no key
                            
                            active_key = keys[0]
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(key=active_key)
                            buffer = buffer.split(start_pattern)[-1]
                            
                        else:
                            if stream_value.find(start_token) != -1: # start token appers in chunk -> pause
                                stream_pause = True 
                                break
                            elif stream_pause:
                                if buffer.find(end_tag) != -1: # if end tag appears in buffer
                                    yield {active_key: buffer.split(end_tag)[0]}
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                    break
                                elif stream_value.find(end_token) != -1: # if ("[blah]" != end_pattern) appeared in buffer 
                                    if buffer.find(end_token + end_token) != -1: # if ]] in buffer -> error
                                        yield False
                                        buffer = buffer.split(end_token + end_token)[-1]
                                        stream_pause = False
                                        break
                                    else:
                                        if buffer.find(start_token + start_token) != -1: # if [[ in buffer -> pause
                                            break
                                        else:
                                            # if [ in buffer (== [blah]) -> stream
                                            yield {active_key: buffer}
                                            buffer = ""
                                            stream_pause = False
                                            break
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield {active_key: buffer}
                                    buffer = ""
                                break
                            
                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(
                        chunk, response_ms, messages, raw_output
                    )
        except Exception as e:
            logger.error(e)
            yield False

    async def __single_type_sp_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator,
        parsing_type: ParsingType,
        start_time: datetime.datetime
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern['start']
            start_fstring = parsing_pattern['start_fstring']
            end_fstring = parsing_pattern['end_fstring']
            start_token = parsing_pattern['start_token']
            end_token = parsing_pattern['end_token']
            
            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            async for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    stream_value : str= chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value 
                    yield stream_value
                    buffer += stream_value
                    
                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield False
                                break
                            
                            if len(keys) == 0:
                                break # no key
                            
                            active_key = keys[0]
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(key=active_key)
                            buffer = buffer.split(start_pattern)[-1]
                            
                        else:
                            if stream_value.find(start_token) != -1: # start token appers in chunk -> pause
                                stream_pause = True 
                                break
                            elif stream_pause:
                                if buffer.find(end_tag) != -1: # if end tag appears in buffer
                                    yield {active_key: buffer.split(end_tag)[0]}
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                elif stream_value.find(end_token) != -1: # if pattern ends  = ("[blah]" != end_pattern) appeared in buffer 
                                    yield False # multiple matching error
                                    stream_pause = False
                                    yield {active_key: buffer}
                                    buffer = ""
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield {active_key: buffer}
                                    buffer = ""
                                break
                            
                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(
                        chunk, response_ms, messages, raw_output
                    )
        except Exception as e:
            logger.error(e)
            yield False