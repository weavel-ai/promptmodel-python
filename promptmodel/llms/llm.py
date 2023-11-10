"""Base module for interacting with LLM APIs."""
import re
import os
import json
import time
import datetime
from typing import Any, AsyncGenerator, List, Dict, Optional, Union, Generator, Tuple
from attr import dataclass

import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import completion, acompletion
from litellm import ModelResponse

from promptmodel.utils.types import LLMResponse, LLMStreamResponse
from promptmodel.utils import logger
from promptmodel.utils.enums import ParsingType, ParsingPattern, get_pattern_by_type
from promptmodel.utils.output_utils import convert_str_to_type, update_dict
from promptmodel.utils.prompt_util import (
    num_tokens_for_messages,
    num_tokens_from_function_call_output,
    num_tokens_from_functions_input,
)

load_dotenv()


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class OpenAIMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = ""
    function_call: Optional[Dict[str, Any]] = None
    name: Optional[str] = None


DEFAULT_MODEL = "gpt-3.5-turbo"


@dataclass
class ParseResult:
    parsed_outputs: Dict[str, Any]
    error: bool
    error_log: Optional[str]


class LLM:
    def __init__(self, rate_limit_manager=None):
        self._rate_limit_manager = rate_limit_manager

    @classmethod
    def __parse_output_pattern__(
        cls, raw_output: str, parsing_type: Optional[ParsingType] = None
    ) -> ParseResult:
        if parsing_type is None:
            return None
        parsing_pattern = get_pattern_by_type(parsing_type)
        whole_pattern = parsing_pattern["whole"]
        parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
        parsed_outputs = {}
        error = False
        error_log: str = None

        try:
            for parsed_result in parsed_results:
                key = parsed_result[0]
                type_str = parsed_result[1]
                value = convert_str_to_type(parsed_result[2], type_str)
                parsed_outputs[key] = value
        except Exception as error:
            error = True
            error_log = str(error)

        return ParseResult(
            parsed_outputs=parsed_outputs,
            error=error,
            error_log=error_log,
        )

    def __validate_openai_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        res = []
        for message in messages:
            res.append(OpenAIMessage(**message))
        return res

    def run(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any] = [],
        model: Optional[str] = DEFAULT_MODEL,
        *args,
        **kwargs,
    ):
        """Return the response from openai chat completion."""
        response = None
        try:
            response = completion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
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
            return LLMResponse(
                api_response=response, raw_output=content, function_call=call_func
            )
        except Exception as e:
            if response:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    async def arun(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any] = [],
        model: Optional[str] = DEFAULT_MODEL,
        *args,
        **kwargs,
    ):
        """Return the response from openai chat completion."""
        response = None
        try:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
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
            return LLMResponse(
                api_response=response, raw_output=content, function_call=call_func
            )
        except Exception as e:
            if response:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    def stream(
        self,
        messages: List[Dict[str, str]],  # input
        functions: List[Any] = [],
        model: Optional[str] = DEFAULT_MODEL,
        *args,
        **kwargs,
    ):
        """Stream openai chat completion."""
        response = None
        try:
            # load_prompt()
            start_time = datetime.datetime.now()
            response = completion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
                functions=functions,
            )
            
            for result in self.__llm_stream_response_generator__(
                    messages, response, start_time, functions
                ):
                    yield result
        except Exception as e:
            return LLMStreamResponse(error=True, error_log=str(e))

    async def astream(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any] = [],
        model: Optional[str] = DEFAULT_MODEL,
        *args,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Parse & stream output from openai chat completion."""
        response = None
        try:
            start_time = datetime.datetime.now()
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
                functions=functions,
            )

            async for result in self.__llm_stream_response_agenerator__(
                    messages, response, start_time, functions
                ):
                    yield result
        except Exception as e:
            yield LLMStreamResponse(error=True, error_log=str(e))

    def run_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
    ) -> Dict[str, str]:
        """Parse and return output from openai chat completion."""
        response = None
        try:
            parse_success = True
            response = completion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=functions,
            )
            raw_output = response.choices[0]["message"]["content"]

            parse_result: Optional[ParseResult] = self.__parse_output_pattern__(
                raw_output, parsing_type
            )
            error_log = parse_result.error_log

            call_func = (
                response.choices[0]["message"]["function_call"]
                if "function_call" in response.choices[0]["message"]
                else None
            )

            if (
                output_keys is not None
                and set(parse_result.parsed_outputs.keys()) != set(output_keys)
                or parse_result.error
            ):
                parse_success = False
                error_log = "Output keys do not match with parsed output keys"

            return LLMResponse(
                api_response=response,
                raw_output=raw_output,
                parsed_outputs=parse_result.parsed_outputs if parse_result else None,
                function_call=call_func,
                error=not parse_success,
                error_log=error_log,
            )
        except Exception as e:
            if response:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    async def arun_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
    ) -> Dict[str, str]:
        """Generate openai chat completion asynchronously, and parse the output.
        Example prompt is as follows:
        -----
        Given a topic, you are required to generate a story.
        You must follow the provided output format.

        Topic:
        {topic}

        Output format:
        [Story]
        ...
        [/Story]

        Now generate the output:
        """
        response = None
        parsed_success = True
        try:
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=functions,
            )
            raw_output = response.choices[0]["message"]["content"]
            parse_result: ParseResult = self.__parse_output_pattern__(
                raw_output, parsing_type
            )
            error_log = parse_result.error_log

            call_func = (
                response.choices[0]["message"]["function_call"]
                if "function_call" in response.choices[0]["message"]
                else None
            )

            if (
                output_keys is not None
                and set(parse_result.parsed_outputs.keys()) != set(output_keys)
                or parse_result.error
            ):
                parse_success = False
                error_log = "Output keys do not match with parsed output keys"

            if (
                output_keys is not None
                and set(parse_result.parsed_outputs.keys()) != set(output_keys)
            ) and parse_success:
                parsed_success = False
                error_log = "Output keys do not match with parsed output keys"

            return LLMResponse(
                api_response=response,
                parsed_outputs=parse_result.parsed_outputs,
                function_call=call_func,
                error=not parsed_success,
                error_log=error_log,
            )
        except Exception as e:
            if response:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    def stream_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Parse & stream output from openai chat completion."""
        response = None
        try:
            if parsing_type == ParsingType.COLON.value:
                # cannot stream colon type
                yield LLMStreamResponse(
                    error=True, error_log="Cannot stream colon type"
                )
                return
            start_time = datetime.datetime.now()
            response = completion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
                functions=functions,
            )

            parsed_outputs = {}
            error_occurs = False
            error_log = ""

            if parsing_type is None:
                for result in self.__llm_stream_response_generator__(
                    messages, response, start_time, functions
                ):
                    yield result

                    if result.error and not error_occurs:
                        error_occurs = True
                        error_log = result.error_log

            elif parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
                for result in self.__double_type_sp_generator__(
                    messages, response, parsing_type, start_time, functions
                ):
                    yield result
                    if result.parsed_outputs:
                        parsed_outputs = update_dict(
                            parsed_outputs, result.parsed_outputs
                        )
                    if result.error and not error_occurs:
                        error_occurs = True
                        error_log = result.error_log
            else:
                for result in self.__single_type_sp_generator__(
                    messages, response, parsing_type, start_time
                ):
                    yield result
                    if result.parsed_outputs:
                        parsed_outputs = update_dict(
                            parsed_outputs, result.parsed_outputs
                        )
                    if result.error and not error_occurs:
                        error_occurs = True
                        error_log = result.error_log

            if (
                output_keys is not None
                and set(parsed_outputs.keys()) != set(output_keys)
            ) and not error_occurs:
                error_occurs = True
                error_log = "Output keys do not match with parsed output keys"
                yield LLMStreamResponse(error=True, error_log=error_log)

        except Exception as e:
            return LLMStreamResponse(error=True, error_log=str(e))

    async def astream_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Parse & stream output from openai chat completion."""
        response = None
        try:
            if parsing_type == ParsingType.COLON.value:
                # cannot stream colon type
                yield LLMStreamResponse(
                    error=True, error_log="Cannot stream colon type"
                )
                return
            start_time = datetime.datetime.now()
            response = await acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                stream=True,
                functions=functions,
            )

            parsed_outputs = {}
            error_occurs = False
            if parsing_type is None:
                async for result in self.__llm_stream_response_agenerator__(
                    messages, response, start_time, functions
                ):
                    yield result

                    if result.error and not error_occurs:
                        error_occurs = True
                        error_log = result.error_log

            elif parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
                async for result in self.__double_type_sp_agenerator__(
                    messages, response, parsing_type, start_time, functions
                ):
                    yield result
                    if result.parsed_outputs:
                        parsed_outputs = update_dict(
                            parsed_outputs, result.parsed_outputs
                        )
                    if result.error and not error_occurs:
                        error_occurs = True
            else:
                async for result in self.__single_type_sp_agenerator__(
                    messages, response, parsing_type, start_time, functions
                ):
                    yield result
                    if result.parsed_outputs:
                        parsed_outputs = update_dict(
                            parsed_outputs, result.parsed_outputs
                        )
                    if result.error and not error_occurs:
                        error_occurs = True

            if (
                output_keys is not None
                and set(parsed_outputs.keys()) != set(output_keys)
            ) and not error_occurs:
                error_occurs = True
                error_log = "Output keys do not match with parsed output keys"
                yield LLMStreamResponse(error=True, error_log=error_log)

        except Exception as e:
            yield LLMStreamResponse(error=True, error_log=str(e))

    def make_model_response(
        self,
        chunk: dict,
        response_ms,
        messages: List[Dict[str, str]],
        raw_output: str,
        function_list: List[Any] = [],
        function_call: Optional[dict] = None,
    ) -> ModelResponse:
        count_start_time = datetime.datetime.now()
        prompt_token: int = num_tokens_for_messages(
            messages=messages, model=chunk["model"]
        )
        completion_token: int = num_tokens_for_messages(
            model=chunk["model"],
            messages=[{"role": "assistant", "content": raw_output}],
        )

        if len(function_list) > 0:
            function_list_token = num_tokens_from_functions_input(
                functions=function_list, model=chunk["model"]
            )
            prompt_token += function_list_token

        if function_call:
            function_call_token = num_tokens_from_function_call_output(
                function_call_output=function_call, model=chunk["model"]
            )
            completion_token += function_call_token

        count_end_time = datetime.datetime.now()
        logger.debug(
            f"counting token time : {(count_end_time - count_start_time).total_seconds() * 1000} ms"
        )

        usage = {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens": prompt_token + completion_token,
        }
        res = ModelResponse(
            id=chunk["id"],
            created=chunk["created"],
            model=chunk["model"],
            usage=usage,
            response_ms=response_ms,
        )
        res["choices"][0]["finish_reason"] = chunk["choices"][0]["finish_reason"]
        res["choices"][0]["message"]["content"] = (
            raw_output if raw_output != "" else None
        )
        res["response_ms"] = response_ms
        if function_call:
            res.choices[0]["message"]["function_call"] = function_call
        return res

    def __double_type_sp_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator,
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern["start"]
            start_fstring = parsing_pattern["start_fstring"]
            end_fstring = parsing_pattern["end_fstring"]
            start_token = parsing_pattern["start_token"]
            end_token = parsing_pattern["end_token"]

            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            function_call = {"name": "", "arguments": ""}
            for chunk in response:
                if (
                    "content" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["content"] is not None
                ):
                    stream_value: str = chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value
                    yield LLMStreamResponse(raw_output=stream_value)
                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield LLMStreamResponse(
                                    error=True,
                                    error_log="Parsing error : Nested key detected",
                                )
                                break
                            if len(keys) == 0:
                                break  # no key
                            active_key, active_type = keys[0]
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(
                                key=active_key, type=active_type
                            )
                            buffer = buffer.split(start_pattern)[-1]

                        else:
                            if (
                                stream_value.find(start_token) != -1
                            ):  # start token appers in chunk -> pause
                                stream_pause = True
                                break
                            elif stream_pause:
                                if (
                                    buffer.find(end_tag) != -1
                                ):  # if end tag appears in buffer
                                    yield LLMStreamResponse(
                                        parsed_outputs={
                                            active_key: buffer.split(end_tag)[0]
                                        }
                                    )
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                    stream_pause = False
                                elif (
                                    stream_value.find(end_token) != -1
                                ):  # if ("[blah]" != end_pattern) appeared in buffer
                                    if (
                                        buffer.find(end_token + end_token) != -1
                                    ):  # if ]] in buffer -> error
                                        yield LLMStreamResponse(
                                            error=True,
                                            error_log="Parsing error : Invalid end tag detected",
                                            parsed_outputs={
                                                active_key: buffer.split(start_token)[0]
                                            },
                                        )
                                        buffer = buffer.split(end_token + end_token)[-1]
                                        stream_pause = False
                                        break
                                    else:
                                        if (
                                            buffer.find(start_token + start_token) != -1
                                        ):  # if [[ in buffer -> pause
                                            break
                                        else:
                                            # if [ in buffer (== [blah]) -> stream
                                            yield LLMStreamResponse(
                                                parsed_outputs={active_key: buffer}
                                            )
                                            buffer = ""
                                            stream_pause = False
                                            break
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield LLMStreamResponse(
                                        parsed_outputs={active_key: buffer}
                                    )
                                    buffer = ""
                                break

                if (
                    "function_call" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["function_call"] is not None
                ):
                    for key, value in chunk["choices"][0]["delta"][
                        "function_call"
                    ].items():
                        function_call[key] += value

                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            function_list=functions,
                            function_call=function_call
                            if chunk["choices"][0]["finish_reason"] == "function_call"
                            else None,
                        ),
                        function_call=function_call
                        if chunk["choices"][0]["finish_reason"] == "function_call"
                        else None,
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    def __llm_stream_response_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator,
        start_time: datetime.datetime,
        functions: List[Any] = [],
    ):
        raw_output = ""
        function_call = {"name": "", "arguments": ""}
        try:
            for chunk in response:
                if (
                    "content" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["content"] is not None
                ):
                    raw_output += chunk["choices"][0]["delta"]["content"]
                    yield LLMStreamResponse(
                        raw_output=chunk["choices"][0]["delta"]["content"]
                    )

                if (
                    "function_call" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["function_call"] is not None
                ):
                    for key, value in chunk["choices"][0]["delta"][
                        "function_call"
                    ].items():
                        function_call[key] += value

                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            function_list=functions,
                            function_call=function_call
                            if chunk["choices"][0]["finish_reason"] == "function_call"
                            else None,
                        ),
                        function_call=function_call
                        if chunk["choices"][0]["finish_reason"] == "function_call"
                        else None,
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    def __single_type_sp_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator,
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern["start"]
            start_fstring = parsing_pattern["start_fstring"]
            end_fstring = parsing_pattern["end_fstring"]
            start_token = parsing_pattern["start_token"]
            end_token = parsing_pattern["end_token"]

            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            function_call = {"name": "", "arguments": ""}
            for chunk in response:
                if (
                    "content" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["content"] is not None
                ):
                    stream_value: str = chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value
                    yield LLMStreamResponse(raw_output=stream_value)
                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield LLMStreamResponse(
                                    error=True,
                                    error_log="Parsing error : Nested key detected",
                                )
                                break
                            if len(keys) == 0:
                                break  # no key

                            active_key, active_type = keys[
                                0
                            ]  # Updated to unpack both key and type
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(
                                key=active_key, type=active_type
                            )
                            buffer = buffer.split(start_pattern)[-1]

                        else:
                            if (
                                stream_value.find(start_token) != -1
                            ):  # start token appers in chunk -> pause
                                stream_pause = True
                                break
                            elif stream_pause:
                                if (
                                    buffer.find(end_tag) != -1
                                ):  # if end tag appears in buffer
                                    yield LLMStreamResponse(
                                        parsed_outputs={
                                            active_key: buffer.split(end_tag)[
                                                0
                                            ].replace(end_tag, "")
                                        }
                                    )
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                    stream_pause = False
                                elif (
                                    stream_value.find(end_token) != -1
                                ):  # if pattern ends  = ("[blah]" != end_pattern) appeared in buffer
                                    if (
                                        active_type == "List"
                                        or active_type == "Dict"
                                        and end_token.find("]") != -1
                                    ):
                                        try:
                                            buffer_dict = json.loads(buffer)
                                            stream_pause = False
                                            continue
                                        except Exception as exception:
                                            logger.error(exception)
                                            yield LLMStreamResponse(
                                                error=True,
                                                error_log="Parsing error : Invalid end tag detected",
                                                parsed_outputs={
                                                    active_key: buffer.split(
                                                        start_token
                                                    )[0]
                                                },
                                            )
                                            stream_pause = False
                                            buffer = ""
                                    yield LLMStreamResponse(
                                        error=True,
                                        error_log="Parsing error : Invalid end tag detected",
                                        parsed_outputs={active_key: buffer},
                                    )
                                    stream_pause = False
                                    buffer = ""
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield LLMStreamResponse(
                                        parsed_outputs={active_key: buffer}
                                    )
                                    buffer = ""
                                break

                if (
                    "function_call" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["function_call"] is not None
                ):
                    for key, value in chunk["choices"][0]["delta"][
                        "function_call"
                    ].items():
                        function_call[key] += value

                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            function_list=functions,
                            function_call=function_call
                            if chunk["choices"][0]["finish_reason"] == "function_call"
                            else None,
                        ),
                        function_call=function_call
                        if chunk["choices"][0]["finish_reason"] == "function_call"
                        else None,
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def __double_type_sp_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator,
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern["start"]
            start_fstring = parsing_pattern["start_fstring"]
            end_fstring = parsing_pattern["end_fstring"]
            start_token = parsing_pattern["start_token"]
            end_token = parsing_pattern["end_token"]

            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            function_call = {"name": "", "arguments": ""}
            async for chunk in response:
                if (
                    "content" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["content"] is not None
                ):
                    stream_value: str = chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value
                    yield LLMStreamResponse(raw_output=stream_value)
                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield LLMStreamResponse(
                                    error=True,
                                    error_log="Parsing error : Nested key detected",
                                )
                                break
                            if len(keys) == 0:
                                break  # no key
                            active_key, active_type = keys[0]
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(
                                key=active_key, type=active_type
                            )
                            buffer = buffer.split(start_pattern)[-1]

                        else:
                            if (
                                stream_value.find(start_token) != -1
                            ):  # start token appers in chunk -> pause
                                stream_pause = True
                                break
                            elif stream_pause:
                                if (
                                    buffer.find(end_tag) != -1
                                ):  # if end tag appears in buffer
                                    yield LLMStreamResponse(
                                        parsed_outputs={
                                            active_key: buffer.split(end_tag)[0]
                                        }
                                    )
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                    stream_pause = False
                                    # break
                                elif (
                                    stream_value.find(end_token) != -1
                                ):  # if ("[blah]" != end_pattern) appeared in buffer
                                    if (
                                        buffer.find(end_token + end_token) != -1
                                    ):  # if ]] in buffer -> error
                                        yield LLMStreamResponse(
                                            error=True,
                                            error_log="Parsing error : Invalid end tag detected",
                                            parsed_outputs={
                                                active_key: buffer.split(start_token)[0]
                                            },
                                        )
                                        buffer = buffer.split(end_token + end_token)[-1]
                                        stream_pause = False
                                        break
                                    else:
                                        if (
                                            buffer.find(start_token + start_token) != -1
                                        ):  # if [[ in buffer -> pause
                                            break
                                        else:
                                            # if [ in buffer (== [blah]) -> stream
                                            yield LLMStreamResponse(
                                                parsed_outputs={active_key: buffer}
                                            )
                                            buffer = ""
                                            stream_pause = False
                                            break
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield LLMStreamResponse(
                                        parsed_outputs={active_key: buffer}
                                    )
                                    buffer = ""
                                break

                if (
                    "function_call" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["function_call"] is not None
                ):
                    for key, value in chunk["choices"][0]["delta"][
                        "function_call"
                    ].items():
                        function_call[key] += value

                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            function_list=functions,
                            function_call=function_call
                            if chunk["choices"][0]["finish_reason"] == "function_call"
                            else None,
                        ),
                        function_call=function_call
                        if chunk["choices"][0]["finish_reason"] == "function_call"
                        else None,
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def __llm_stream_response_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator,
        start_time: datetime.datetime,
        functions: List[Any] = [],
    ):
        raw_output = ""
        function_call = {"name": "", "arguments": ""}
        try:
            async for chunk in response:
                if (
                    "content" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["content"] is not None
                ):
                    raw_output += chunk["choices"][0]["delta"]["content"]
                    yield LLMStreamResponse(
                        raw_output=chunk["choices"][0]["delta"]["content"]
                    )

                if (
                    "function_call" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["function_call"] is not None
                ):
                    for key, value in chunk["choices"][0]["delta"][
                        "function_call"
                    ].items():
                        function_call[key] += value

                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            function_list=functions,
                            function_call=function_call
                            if chunk["choices"][0]["finish_reason"] == "function_call"
                            else None,
                        ),
                        function_call=function_call
                        if chunk["choices"][0]["finish_reason"] == "function_call"
                        else None,
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def __single_type_sp_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator,
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
    ):
        try:
            parsing_pattern = get_pattern_by_type(parsing_type)
            start_tag = parsing_pattern["start"]
            start_fstring = parsing_pattern["start_fstring"]
            end_fstring = parsing_pattern["end_fstring"]
            start_token = parsing_pattern["start_token"]
            end_token = parsing_pattern["end_token"]

            buffer = ""
            raw_output = ""
            active_key = None
            stream_pause = False
            end_tag = None
            function_call = {"name": "", "arguments": ""}
            async for chunk in response:
                if (
                    "content" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["content"] is not None
                ):
                    stream_value: str = chunk["choices"][0]["delta"]["content"]
                    raw_output += stream_value
                    yield LLMStreamResponse(raw_output=stream_value)
                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            if len(keys) > 1:
                                yield LLMStreamResponse(
                                    error=True,
                                    error_log="Parsing error : Nested key detected",
                                )
                                break
                            if len(keys) == 0:
                                break  # no key

                            active_key, active_type = keys[
                                0
                            ]  # Updated to unpack both key and type
                            end_tag = end_fstring.format(key=active_key)
                            # delete start tag from buffer
                            start_pattern = start_fstring.format(
                                key=active_key, type=active_type
                            )
                            buffer = buffer.split(start_pattern)[-1]

                        else:
                            if (
                                stream_value.find(start_token) != -1
                            ):  # start token appers in chunk -> pause
                                stream_pause = True
                                break
                            elif stream_pause:
                                if (
                                    buffer.find(end_tag) != -1
                                ):  # if end tag appears in buffer
                                    yield LLMStreamResponse(
                                        parsed_outputs={
                                            active_key: buffer.split(end_tag)[
                                                0
                                            ].replace(end_tag, "")
                                        }
                                    )
                                    buffer = buffer.split(end_tag)[-1]
                                    active_key = None
                                    stream_pause = False
                                elif (
                                    stream_value.find(end_token) != -1
                                ):  # if pattern ends  = ("[blah]" != end_pattern) appeared in buffer
                                    if (
                                        active_type == "List"
                                        or active_type == "Dict"
                                        and end_token.find("]") != -1
                                    ):
                                        try:
                                            buffer_dict = json.loads(buffer)
                                            stream_pause = False
                                            continue
                                        except Exception as exception:
                                            logger.error(exception)
                                            yield LLMStreamResponse(
                                                error=True,
                                                error_log="Parsing error : Invalid end tag detected",
                                                parsed_outputs={
                                                    active_key: buffer.split(
                                                        start_token
                                                    )[0]
                                                },
                                            )
                                            stream_pause = False
                                            buffer = ""
                                    yield LLMStreamResponse(
                                        error=True,
                                        error_log="Parsing error : Invalid end tag detected",
                                        parsed_outputs={active_key: buffer},
                                    )
                                    stream_pause = False
                                    buffer = ""
                                break
                            else:
                                # no start token, no stream_pause (not inside of tag)
                                if buffer:
                                    yield LLMStreamResponse(
                                        parsed_outputs={active_key: buffer}
                                    )
                                    buffer = ""
                                break

                if (
                    "function_call" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["function_call"] is not None
                ):
                    for key, value in chunk["choices"][0]["delta"][
                        "function_call"
                    ].items():
                        function_call[key] += value

                if chunk["choices"][0]["finish_reason"] != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            function_list=functions,
                            function_call=function_call
                            if chunk["choices"][0]["finish_reason"] == "function_call"
                            else None,
                        ),
                        function_call=function_call
                        if chunk["choices"][0]["finish_reason"] == "function_call"
                        else None,
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))
