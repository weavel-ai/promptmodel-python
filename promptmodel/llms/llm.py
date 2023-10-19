"""Base module for interacting with LLM APIs."""
import re
import os
import json
import time
import datetime
from typing import Any, AsyncGenerator, List, Dict, Optional, Union, Generator

import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import completion, acompletion
from litellm import ModelResponse, RateLimitManager
from litellm.utils import prompt_token_calculator, token_counter

from promptmodel.utils import logger

load_dotenv()


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class OpenAIMessage(BaseModel):
    role: str
    content: str


class LLM:
    def __init__(
        self, rate_limit_manager: Optional[RateLimitManager] = None
    ):
        self._model: str
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

    def __validate_openai_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        return [OpenAIMessage(**message) for message in messages]

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        _model = model or self._model
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        )
        res = response.choices[0]["message"]["content"]
        if show_response:
            return res, response
        return res
    
    def generate_function_call(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any],
        model: Optional[str] = None,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        _model = model or self._model
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            function_call=True,
            functions=functions
        )
        content = response.choices[0]["message"]["content"]
        call_func = response.choices[0]["message"]["function_call"] if "function_call" in response.choices[0]["message"] else None
        if show_response:
            return content, call_func, response
        return content, call_func

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        _model = model or self._model
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        else:
            response = await acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        res = response.choices[0]["message"]["content"]
        if show_response:
            return res, response
        return res

    async def agenerate_function_call(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any],
        model: Optional[str] = None,
        show_response: bool = False,
    ):
        """Return the response from openai chat completion."""
        _model = model or self._model
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                function_call=True,
                functions=functions
            )
        else:
            response = await acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                function_call=True,
                functions=functions
            )
        content = response.choices[0]["message"]["content"] if "content" in response.choices[0]["message"] else None
        call_func = response.choices[0]["message"]["function_call"] if "function_call" in response.choices[0]["message"] else None
        if show_response:
            return content, call_func, response
        return content, call_func

    def stream(
        self,
        messages: List[Dict[str, str]],  # input
        model: Optional[str] = None,
        show_response: bool = False,
    ):
        """Stream openai chat completion."""
        _model = model or self._model
        # load_prompt()
        start_time =  datetime.datetime.now()
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True
        ).choices[0]["message"]["content"]
        
        raw_output = ""
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                raw_output += chunk["choices"][0]["delta"]["content"]
                yield chunk["choices"][0]["delta"]["content"]
            if chunk['choices'][0]['finish_reason'] != None:
                if show_response:
                    end_time =  datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(chunk, response_ms, messages, raw_output)

    def generate_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
        show_response: bool = False,
    ) -> Dict[str, str]:
        """Parse and return output from openai chat completion."""
        _model = model or self._model
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        )
        raw_output = response.choices[0]["message"]["content"]

        parsed_output = {}
        for key in output_keys:
            parsed_output[key] = self.__parse_output__(raw_output, key)

        if show_response:
            return parsed_output, response
        return parsed_output

    def stream_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
        show_response: bool = False,
        **kwargs,
    ) -> Generator[Dict[str, str], None, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        start_time =  datetime.datetime.now()
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )

        cache = ""
        for chunk in response:
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
                if key not in output_keys:  # 미리 정해둔 output key가 아님
                    continue
                if stream_value.find("]") != -1 or "[" in re.sub(
                    r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                    "",
                    stripped_output,
                    flags=re.DOTALL,
                ):  # 현재 stream 중인 output이 [[key end]] 부분일 경우에는 pause_stream을 True로 설정
                    if stream_value.find("[") != -1:
                        if cache.find("[[") != -1:
                            pause_stream = True
                        else:
                            cache += "["
                    pause_stream = True
                if pause_stream:
                    if stream_value.find("]") != -1:
                        cache = ""
                        pause_stream = False
                    continue

                yield {key: stream_value}
            if chunk['choices'][0]['finish_reason'] != None:
                if show_response:
                    end_time =  datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(chunk, response_ms, messages, raw_output)

    async def agenerate_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
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
        _model = model or self._model
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        else:
            response = await acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
            )
        raw_output = response.choices[0]["message"]["content"]
        logger.debug(f"Output:\n{raw_output}")
        parsed_output = {}
        for key in output_keys:
            output = self.__parse_output__(raw_output, key)
            if output:
                parsed_output[key] = output

        if show_response:
            return parsed_output, response
        return parsed_output

    def generate_and_parse_function_call(
        self,
        messages: List[Dict[str, str]],
        function_list: [],
        model: Optional[str] = "gpt-3.5-turbo-0613",
        show_response: bool = False,
    ) -> Generator[str, None, None]:
        """
        Parse by function call arguments
        """
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            functions=function_list,
            function_call="auto",
        )
        print(response)
        function_args = response["choices"][0]["message"]["function_call"]["arguments"]
        # make function_args to dict
        function_args = function_args.replace("'", '"')
        function_args = json.loads(function_args)
        
        if show_response:
            return function_args, response
        return function_args

    async def agenerate_and_parse_function_call(
        self,
        messages: List[Dict[str, str]],
        function_list: [],
        model: Optional[str] = "gpt-3.5-turbo-0613",
        show_response: bool = False,
    ) -> Generator[str, None, None]:
        """
        Parse by function call arguments
        """
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=function_list,
                function_call="auto",
            )
        else:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=function_list,
                function_call="auto",
            )
        function_args = response["choices"][0]["message"]["function_call"]["arguments"]
        # make function_args to dict
        function_args = function_args.replace("'", '"')
        function_args = json.loads(function_args)
        
        if show_response:
            return function_args, response
        return function_args

    async def astream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        show_response: bool = False,
    ) -> Generator[Dict[str, str], None, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        start_time = datetime.datetime.now()
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
            )
        else:
            response = await acompletion(
                model=_model,
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
            if chunk['choices'][0]['finish_reason'] != None:
                if show_response:
                    end_time =  datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(chunk, response_ms, messages, raw_output)

    async def astream_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
        show_response: bool = False
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        start_time = datetime.datetime.now()
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
            )
        else:
            response = await acompletion(
                model=_model,
                messages=[
                    message.model_dump()
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
            )

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
                if key not in output_keys:  # 미리 정해둔 output key가 아님
                    continue
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
                    yield {key: stream_value}
                elif stream_value.find("]") != -1:
                    # Current stream_value (that includes ]) isn't yielded, but the next stream_values will be yielded.
                    cache = ""
                    pause_stream = False
            if chunk['choices'][0]['finish_reason'] != None:
                if show_response:
                    end_time =  datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield self.make_model_response(chunk, response_ms, messages, raw_output)

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

    async def astream_and_parse_function_call(
        self,
        messages: List[Dict[str, str]],
        function_list: [],
        output_key: str,
        model: Optional[str] = "gpt-3.5-turbo-0613",
        show_response: bool = False,
    ) -> Generator[str, None, None]:
        start_time = datetime.datetime.now()
        if self._rate_limit_manager:
            response = await self._rate_limit_manager.acompletion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            functions=function_list,
            function_call="auto",
            stream=True,
            )
        else:
            response = await acompletion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            functions=function_list,
            function_call="auto",
            stream=True,
            )   

        function_call = {
            "name" : "",
            "arguments" : ""
        }
        function_args = ""
        start_to_stream = False
        raw_output = ""
        async for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                raw_output += chunk["choices"][0]["delta"]["content"]
            if "function_call" in chunk["choices"][0]["delta"]:
                if "name" in chunk["choices"][0]["delta"]["function_call"]:
                    function_name = chunk["choices"][0]["delta"]["function_call"][
                        "name"
                    ]
                    function_call['name'] += function_name
                
                function_args += chunk["choices"][0]["delta"]["function_call"][
                    "arguments"
                ]
                if f'"{output_key}":' in function_args:
                    if not start_to_stream:
                        start_to_stream = True
                        # yield function_args without output_key
                        yield {
                            output_key: function_args.replace(
                                f'"{output_key}":', ""
                            )
                        }
                    else:
                        yield {
                            output_key: chunk["choices"][0]["delta"][
                                "function_call"
                            ]["arguments"]
                        }
            if chunk['choices'][0]['finish_reason'] != None:
                if show_response:
                    end_time =  datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    function_call['arguments'] = function_args
                    yield self.make_model_response(chunk, response_ms, messages, raw_output, function_call)
    
    
    def make_model_response(
        chunk: dict, response_ms, messages: List[Dict[str, str]], raw_output: str, function_call: Optional[dict]
    ) -> ModelResponse:
        choices = [
            {
                "index": 0,
                "message" : {
                    "role": "assistant",
                    "content": raw_output
                },
                "finish_reason" : chunk['choices'][0]['finish_reason']
            }
        ]
        if function_call:
            choices[0]['message']['function_call'] = function_call
        prompt_token: int = prompt_token_calculator(chunk['model'], messages)
        completion_token: int = token_counter(chunk['model'], raw_output)
        usage = {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens" : prompt_token + completion_token
        }
        res = ModelResponse(
            id=chunk['id'],
            choices=choices,
            created=chunk['created'],
            model=chunk['model'],
            usage=usage,
            response_ms=response_ms,
        )
        return res