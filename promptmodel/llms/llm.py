"""Base module for interacting with LLM APIs."""
import re
import json
import datetime
from typing import Any, AsyncGenerator, List, Dict, Optional, Generator
from attr import dataclass

from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import completion, acompletion

from promptmodel.types.response import (
    LLMResponse,
    LLMStreamResponse,
    ModelResponse,
    Usage,
    Choices,
    Message,
    FunctionCall,
    ChatCompletionMessageToolCall,
    Function,
    ChoiceDeltaToolCallFunction,
    ChoiceDeltaToolCall,
)
from promptmodel.utils import logger
from promptmodel.types.enums import ParsingType, ParsingPattern, get_pattern_by_type
from promptmodel.utils.output_utils import convert_str_to_type, update_dict
from promptmodel.utils.token_counting import (
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
    def __init__(self):
        pass

    @classmethod
    def __parse_output_pattern__(
        cls,
        raw_output: Optional[str] = None,
        parsing_type: Optional[ParsingType] = None,
    ) -> ParseResult:
        if parsing_type is None:
            return ParseResult(parsed_outputs={}, error=False, error_log=None)
        if raw_output is None:
            return ParseResult(parsed_outputs={}, error=True, error_log="No content")
        parsing_pattern = get_pattern_by_type(parsing_type)
        whole_pattern = parsing_pattern["whole"]
        parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
        parsed_outputs = {}
        error: bool = False
        error_log: str = None

        try:
            for parsed_result in parsed_results:
                key = parsed_result[0]
                type_str = parsed_result[1]
                value = convert_str_to_type(parsed_result[2], type_str)
                parsed_outputs[key] = value
        except Exception as e:
            error = True
            error_log = str(e)

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
        tools: Optional[List[Any]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Return the response from openai chat completion."""
        response = None
        try:
            response: ModelResponse = completion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=functions,
                tools=tools,
                api_key=api_key,
            )

            content: Optional[str] = getattr(
                response.choices[0].message, "content", None
            )

            call_func: Optional[FunctionCall] = getattr(
                response.choices[0].message, "function_call", None
            )

            call_tools: Optional[List[ChatCompletionMessageToolCall]] = getattr(
                response.choices[0].message, "tool_calls", None
            )

            return LLMResponse(
                api_response=response,
                raw_output=content,
                function_call=call_func if call_func else None,
                tool_calls=call_tools if call_tools else None,
            )
        except Exception as e:
            if response is not None:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    async def arun(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Return the response from openai chat completion."""
        response = None
        try:
            response: ModelResponse = await acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=functions,
                tools=tools,
                api_key=api_key,
            )
            content: Optional[str] = getattr(
                response.choices[0].message, "content", None
            )

            call_func: Optional[FunctionCall] = getattr(
                response.choices[0].message, "function_call", None
            )

            call_tools: Optional[ChatCompletionMessageToolCall] = getattr(
                response.choices[0].message, "tool_calls", None
            )

            return LLMResponse(
                api_response=response,
                raw_output=content,
                function_call=call_func if call_func else None,
                tool_calls=call_tools if call_tools else None,
            )

        except Exception as e:
            if response is not None:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    def stream(
        self,
        messages: List[Dict[str, str]],  # input
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
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
                tools=tools,
                api_key=api_key,
            )

            for chunk in self.__llm_stream_response_generator__(
                messages, response, start_time, functions, tools
            ):
                yield chunk
        except Exception as e:
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def astream(
        self,
        messages: List[Dict[str, str]],
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
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
                tools=tools,
                api_key=api_key,
            )

            async for chunk in self.__llm_stream_response_agenerator__(
                messages, response, start_time, functions, tools
            ):
                yield chunk
        except Exception as e:
            yield LLMStreamResponse(error=True, error_log=str(e))

    def run_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        """Parse and return output from openai chat completion."""
        response = None
        parsed_success = True
        parse_result = None
        error_log = None
        try:
            response: ModelResponse = completion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=functions,
                tools=tools,
                api_key=api_key,
            )
            raw_output = getattr(response.choices[0].message, "content", None)

            call_func: Optional[FunctionCall] = getattr(
                response.choices[0].message, "function_call", None
            )

            call_tools: Optional[List[ChatCompletionMessageToolCall]] = getattr(
                response.choices[0].message, "tool_calls", None
            )

            if not call_func and not call_tools:
                # function call does not appear in output

                parse_result: ParseResult = self.__parse_output_pattern__(
                    raw_output, parsing_type
                )

                # if output_keys exist & parsed_outputs does not match with output_keys -> error
                # if parse_result.error -> error
                if (
                    output_keys is not None
                    and set(parse_result.parsed_outputs.keys()) != set(output_keys)
                ) or parse_result.error:
                    parsed_success = False
                    error_log = (
                        "Output keys do not match with parsed output keys"
                        if not parse_result.error_log
                        else parse_result.error_log
                    )

            return LLMResponse(
                api_response=response,
                raw_output=raw_output,
                parsed_outputs=parse_result.parsed_outputs if parse_result else None,
                function_call=call_func if call_func else None,
                tool_calls=call_tools if call_tools else None,
                error=not parsed_success,
                error_log=error_log,
            )
        except Exception as e:
            if response is not None:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    async def arun_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
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
        parse_result = None
        error_log = None
        try:
            response: ModelResponse = await acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True)
                    for message in self.__validate_openai_messages(messages)
                ],
                functions=functions,
                tools=tools,
                api_key=api_key,
            )
            raw_output = getattr(response.choices[0].message, "content", None)

            call_func: Optional[FunctionCall] = getattr(
                response.choices[0].message, "function_call", None
            )

            call_tools: Optional[List[ChatCompletionMessageToolCall]] = getattr(
                response.choices[0].message, "tool_calls", None
            )

            if not call_func and not call_tools:
                # function call does not appear in output
                parse_result: ParseResult = self.__parse_output_pattern__(
                    raw_output, parsing_type
                )

                # if output_keys exist & parsed_outputs does not match with output_keys -> error
                # if parse_result.error -> error
                if (
                    output_keys is not None
                    and set(parse_result.parsed_outputs.keys()) != set(output_keys)
                ) or parse_result.error:
                    parsed_success = False
                    error_log = (
                        "Output keys do not match with parsed output keys"
                        if not parse_result.error_log
                        else parse_result.error_log
                    )

            return LLMResponse(
                api_response=response,
                raw_output=raw_output,
                parsed_outputs=parse_result.parsed_outputs if parse_result else None,
                function_call=call_func if call_func else None,
                tool_calls=call_tools if call_tools else None,
                error=not parsed_success,
                error_log=error_log,
            )
        except Exception as e:
            if response is not None:
                return LLMResponse(api_response=response, error=True, error_log=str(e))
            else:
                return LLMResponse(api_response=None, error=True, error_log=str(e))

    def stream_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
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
                tools=tools,
                api_key=api_key,
            )

            parsed_outputs = {}
            error_occurs = False
            error_log = None

            if len(functions) > 0 or (tools and len(tools) > 0):
                # if function exists, cannot parsing in stream time
                # just stream raw output and parse after stream
                streamed_outputs = {
                    "content": "",
                    "function_call": None,
                    "api_response": None,
                }
                response_with_api_res = None
                for chunk in self.__llm_stream_response_generator__(
                    messages, response, start_time, functions, tools
                ):
                    if chunk.raw_output:
                        streamed_outputs["content"] += chunk.raw_output
                    if chunk.function_call:
                        streamed_outputs["function_call"] = chunk.function_call
                    if (
                        chunk.api_response
                        and getattr(chunk.api_response.choices[0], "delta", None)
                        is None
                    ):  # only get the last api_response, not delta response
                        streamed_outputs["api_response"] = chunk.api_response
                        response_with_api_res = chunk
                    else:
                        yield chunk

                    if chunk.error and not error_occurs:
                        error_occurs = True
                        error_log = chunk.error_log

                if not streamed_outputs["function_call"]:
                    # if function call does not exist in output
                    # able to parse
                    parse_result: ParseResult = self.__parse_output_pattern__(
                        streamed_outputs["content"], parsing_type
                    )

                    error_occurs = parse_result.error or error_occurs
                    error_log = parse_result.error_log if not error_log else error_log

                    if (
                        output_keys is not None
                        and set(parse_result.parsed_outputs.keys()) != set(output_keys)
                    ) or error_occurs:
                        error_occurs = True
                        error_log = (
                            "Output keys do not match with parsed output keys"
                            if not error_log
                            else error_log
                        )
                        yield LLMStreamResponse(
                            api_response=streamed_outputs["api_response"],
                            error=True,
                            error_log=error_log,
                        )
                    else:
                        response_with_api_res.parsed_outputs = (
                            parse_result.parsed_outputs
                        )
                        yield response_with_api_res
                else:
                    yield response_with_api_res
            else:
                if parsing_type is None:
                    for chunk in self.__llm_stream_response_generator__(
                        messages, response, start_time, functions, tools
                    ):
                        yield chunk

                        if chunk.error and not error_occurs:
                            error_occurs = True
                            error_log = chunk.error_log

                elif parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
                    for chunk in self.__double_type_sp_generator__(
                        messages, response, parsing_type, start_time, functions, tools
                    ):
                        yield chunk
                        if chunk.parsed_outputs:
                            parsed_outputs = update_dict(
                                parsed_outputs, chunk.parsed_outputs
                            )
                        if chunk.error and not error_occurs:
                            error_occurs = True
                            error_log = chunk.error_log
                else:
                    for chunk in self.__single_type_sp_generator__(
                        messages, response, parsing_type, start_time, functions, tools
                    ):
                        yield chunk
                        if chunk.parsed_outputs:
                            parsed_outputs = update_dict(
                                parsed_outputs, chunk.parsed_outputs
                            )
                        if chunk.error and not error_occurs:
                            error_occurs = True
                            error_log = chunk.error_log

                if (
                    output_keys is not None
                    and set(parsed_outputs.keys()) != set(output_keys)
                ) and not error_occurs:
                    error_occurs = True
                    error_log = "Output keys do not match with parsed output keys"
                    yield LLMStreamResponse(error=True, error_log=error_log)

        except Exception as e:
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def astream_and_parse(
        self,
        messages: List[Dict[str, str]],
        parsing_type: Optional[ParsingType] = None,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
        output_keys: Optional[List[str]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        api_key: Optional[str] = None,
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
                tools=tools,
                api_key=api_key,
            )

            parsed_outputs = {}
            error_occurs = False  # error in stream time
            error_log = None
            if len(functions) > 0 or (tools and len(tools) > 0):
                # if function exists, cannot parsing in stream time
                # just stream raw output and parse after stream
                streamed_outputs = {
                    "content": "",
                    "function_call": None,
                    "api_response": None,
                }
                response_with_api_res = None
                async for chunk in self.__llm_stream_response_agenerator__(
                    messages, response, start_time, functions, tools
                ):
                    if chunk.raw_output:
                        streamed_outputs["content"] += chunk.raw_output
                    if chunk.function_call:
                        streamed_outputs["function_call"] = chunk.function_call
                    if (
                        chunk.api_response
                        and getattr(chunk.api_response.choices[0], "delta", None)
                        is None
                    ):
                        streamed_outputs["api_response"] = chunk.api_response
                        response_with_api_res = chunk
                    else:
                        yield chunk

                    if chunk.error and not error_occurs:
                        error_occurs = True
                        error_log = chunk.error_log

                if not streamed_outputs["function_call"]:
                    # if function call does not exist in output
                    # able to parse
                    parse_result: ParseResult = self.__parse_output_pattern__(
                        streamed_outputs["content"], parsing_type
                    )

                    error_occurs = parse_result.error or error_occurs
                    error_log = parse_result.error_log if not error_log else error_log
                    if (
                        output_keys is not None
                        and set(parse_result.parsed_outputs.keys()) != set(output_keys)
                    ) or error_occurs:
                        error_occurs = True
                        error_log = (
                            "Output keys do not match with parsed output keys"
                            if not error_log
                            else error_log
                        )
                        yield LLMStreamResponse(
                            api_response=streamed_outputs["api_response"],
                            error=True,
                            error_log=error_log,
                        )
                    else:
                        response_with_api_res.parsed_outputs = (
                            parse_result.parsed_outputs
                        )
                        yield response_with_api_res
                else:
                    yield response_with_api_res
            else:
                if parsing_type is None:
                    async for chunk in self.__llm_stream_response_agenerator__(
                        messages, response, start_time, functions, tools
                    ):
                        yield chunk

                        if chunk.error and not error_occurs:
                            error_occurs = True
                            error_log = chunk.error_log

                elif parsing_type == ParsingType.DOUBLE_SQUARE_BRACKET.value:
                    async for chunk in self.__double_type_sp_agenerator__(
                        messages, response, parsing_type, start_time, functions, tools
                    ):
                        yield chunk
                        if chunk.parsed_outputs:
                            parsed_outputs = update_dict(
                                parsed_outputs, chunk.parsed_outputs
                            )
                        if chunk.error and not error_occurs:
                            error_occurs = True
                else:
                    async for chunk in self.__single_type_sp_agenerator__(
                        messages, response, parsing_type, start_time, functions, tools
                    ):
                        yield chunk
                        if chunk.parsed_outputs:
                            parsed_outputs = update_dict(
                                parsed_outputs, chunk.parsed_outputs
                            )
                        if chunk.error and not error_occurs:
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
        chunk: ModelResponse,
        response_ms,
        messages: List[Dict[str, str]],
        raw_output: str,
        functions: List[Any] = [],
        function_call: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        count_start_time = datetime.datetime.now()
        prompt_token: int = num_tokens_for_messages(
            messages=messages, model=chunk["model"]
        )
        completion_token: int = num_tokens_for_messages(
            model=chunk["model"],
            messages=[{"role": "assistant", "content": raw_output}],
        )

        if len(functions) > 0 or (tools and len(tools) > 0):
            functions_token = num_tokens_from_functions_input(
                functions=functions, model=chunk["model"]
            )
            prompt_token += functions_token

        if tools and len(tools) > 0:
            tools_token = num_tokens_from_functions_input(
                functions=[tool["function"] for tool in tools], model=chunk["model"]
            )
            prompt_token += tools_token

        # if function_call:
        #     function_call_token = num_tokens_from_function_call_output(
        #         function_call_output=function_call, model=chunk["model"]
        #     )
        #     completion_token += function_call_token

        count_end_time = datetime.datetime.now()
        logger.debug(
            f"counting token time : {(count_end_time - count_start_time).total_seconds() * 1000} ms"
        )

        usage = Usage(
            **{
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": prompt_token + completion_token,
            }
        )

        last_message = Message(
            role=chunk.choices[0].delta.role
            if getattr(chunk.choices[0].delta, "role", None)
            else "assistant",
            content=raw_output if raw_output != "" else None,
            function_call=function_call if function_call else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        choices = [
            Choices(finish_reason=chunk.choices[0].finish_reason, message=last_message)
        ]

        res = ModelResponse(
            id=chunk["id"],
            created=chunk["created"],
            model=chunk["model"],
            stream=True,
        )
        res.choices = choices
        res.usage = usage
        res._response_ms = response_ms
        return res

    def __llm_stream_response_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator[ModelResponse, None, None],
        start_time: datetime.datetime,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        raw_output = ""
        function_call = {"name": "", "arguments": ""}
        tool_calls = []

        try:
            for chunk in response:
                yield_api_response_with_fc = False
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    for key, value in (
                        chunk.choices[0].delta.function_call.model_dump().items()
                    ):
                        if value is not None:
                            function_call[key] += value

                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    # tool_calls: list
                    tool_calls_delta: List[Any] = chunk.choices[0].delta.tool_calls
                    index = tool_calls_delta[0].index
                    if index == len(tool_calls):
                        tool_calls.append(
                            {
                                "id": tool_calls_delta[0].id,
                                "function": {},
                                "type": "function",
                            }
                        )
                    tool_delta: ChoiceDeltaToolCallFunction = tool_calls_delta[
                        0
                    ].function
                    tool_calls[index]["function"] = update_dict(
                        tool_calls[index]["function"], tool_delta.model_dump()
                    )

                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    raw_output += chunk.choices[0].delta.content
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=chunk.choices[0].delta.content,
                    )

                if chunk.choices[0].finish_reason != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=functions,
                            function_call=function_call
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=tools,
                            tool_calls=tool_calls
                            if chunk.choices[0].finish_reason == "tool_calls"
                            else None,
                        )
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    def __single_type_sp_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator[ModelResponse, None, None],
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
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
            tool_calls = []

            for chunk in response:
                yield_api_response_with_fc = False
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    for key, value in (
                        chunk.choices[0].delta.function_call.model_dump().items()
                    ):
                        if value is not None:
                            function_call[key] += value

                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    # tool_calls: list
                    tool_calls_delta: List[Any] = chunk.choices[0].delta.tool_calls
                    index = tool_calls_delta[0].index
                    if index == len(tool_calls):
                        tool_calls.append(
                            {
                                "id": tool_calls_delta[0].id,
                                "function": {},
                                "type": "function",
                            }
                        )
                    tool_delta: ChoiceDeltaToolCallFunction = tool_calls_delta[
                        0
                    ].function
                    tool_calls[index]["function"] = update_dict(
                        tool_calls[index]["function"], tool_delta.model_dump()
                    )

                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    stream_value: str = chunk.choices[0].delta.content
                    raw_output += stream_value
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=stream_value,
                    )

                    buffer += stream_value
                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
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

                if chunk.choices[0].finish_reason != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=functions,
                            function_call=function_call
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=tools,
                            tool_calls=tool_calls
                            if chunk.choices[0].finish_reason == "tool_calls"
                            else None,
                        )
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    def __double_type_sp_generator__(
        self,
        messages: List[Dict[str, str]],
        response: Generator[ModelResponse, None, None],
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
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
            tool_calls = []

            for chunk in response:
                yield_api_response_with_fc = False
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    for key, value in (
                        chunk.choices[0].delta.function_call.model_dump().items()
                    ):
                        if value is not None:
                            function_call[key] += value

                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    # tool_calls: list
                    tool_calls_delta: List[Any] = chunk.choices[0].delta.tool_calls
                    index = tool_calls_delta[0].index
                    if index == len(tool_calls):
                        tool_calls.append(
                            {
                                "id": tool_calls_delta[0].id,
                                "function": {},
                                "type": "function",
                            }
                        )
                    tool_delta: ChoiceDeltaToolCallFunction = tool_calls_delta[
                        0
                    ].function
                    tool_calls[index]["function"] = update_dict(
                        tool_calls[index]["function"], tool_delta.model_dump()
                    )

                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    stream_value: str = chunk.choices[0].delta.content
                    raw_output += stream_value
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=stream_value,
                    )

                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
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

                if chunk.choices[0].finish_reason != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=functions,
                            function_call=function_call
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=tools,
                            tool_calls=tool_calls
                            if chunk.choices[0].finish_reason == "tool_calls"
                            else None,
                        )
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def __llm_stream_response_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator[ModelResponse, None],
        start_time: datetime.datetime,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        raw_output = ""
        function_call = {"name": "", "arguments": ""}
        tool_calls = []
        try:
            async for chunk in response:
                yield_api_response_with_fc = False
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    for key, value in (
                        chunk.choices[0].delta.function_call.model_dump().items()
                    ):
                        if value is not None:
                            function_call[key] += value

                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    # tool_calls: list
                    tool_calls_delta: List[Any] = chunk.choices[0].delta.tool_calls
                    index = tool_calls_delta[0].index
                    if index == len(tool_calls):
                        tool_calls.append(
                            {
                                "id": tool_calls_delta[0].id,
                                "function": {},
                                "type": "function",
                            }
                        )
                    tool_delta: ChoiceDeltaToolCallFunction = tool_calls_delta[
                        0
                    ].function
                    tool_calls[index]["function"] = update_dict(
                        tool_calls[index]["function"], tool_delta.model_dump()
                    )

                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    stream_value: str = chunk.choices[0].delta.content
                    raw_output += stream_value
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=stream_value,
                    )

                if chunk.choices[0].finish_reason != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=functions,
                            function_call=function_call
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=tools,
                            tool_calls=tool_calls
                            if chunk.choices[0].finish_reason == "tool_calls"
                            else None,
                        )
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def __single_type_sp_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator[ModelResponse, None],
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
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
            tool_calls = []

            async for chunk in response:
                yield_api_response_with_fc = False
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    for key, value in (
                        chunk.choices[0].delta.function_call.model_dump().items()
                    ):
                        if value is not None:
                            function_call[key] += value

                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    # tool_calls: list
                    tool_calls_delta: List[Any] = chunk.choices[0].delta.tool_calls
                    index = tool_calls_delta[0].index
                    if index == len(tool_calls):
                        tool_calls.append(
                            {
                                "id": tool_calls_delta[0].id,
                                "function": {},
                                "type": "function",
                            }
                        )
                    tool_delta: ChoiceDeltaToolCallFunction = tool_calls_delta[
                        0
                    ].function
                    tool_calls[index]["function"] = update_dict(
                        tool_calls[index]["function"], tool_delta.model_dump()
                    )

                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    stream_value: str = chunk.choices[0].delta.content
                    raw_output += stream_value
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=stream_value,
                    )

                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
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

                if chunk.choices[0].finish_reason != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=functions,
                            function_call=function_call
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=tools,
                            tool_calls=tool_calls
                            if chunk.choices[0].finish_reason == "tool_calls"
                            else None,
                        )
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))

    async def __double_type_sp_agenerator__(
        self,
        messages: List[Dict[str, str]],
        response: AsyncGenerator[ModelResponse, None],
        parsing_type: ParsingType,
        start_time: datetime.datetime,
        functions: List[Any] = [],
        tools: Optional[List[Any]] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
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
            tool_calls = []

            async for chunk in response:
                yield_api_response_with_fc = False
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    for key, value in (
                        chunk.choices[0].delta.function_call.model_dump().items()
                    ):
                        if value is not None:
                            function_call[key] += value

                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    # tool_calls: list
                    tool_calls_delta: List[Any] = chunk.choices[0].delta.tool_calls
                    index = tool_calls_delta[0].index
                    if index == len(tool_calls):
                        tool_calls.append(
                            {
                                "id": tool_calls_delta[0].id,
                                "function": {},
                                "type": "function",
                            }
                        )
                    tool_delta: ChoiceDeltaToolCallFunction = tool_calls_delta[
                        0
                    ].function
                    tool_calls[index]["function"] = update_dict(
                        tool_calls[index]["function"], tool_delta.model_dump()
                    )

                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    stream_value: str = chunk.choices[0].delta.content
                    raw_output += stream_value
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=stream_value,
                    )

                    buffer += stream_value

                    while True:
                        if active_key is None:
                            keys = re.findall(start_tag, buffer, flags=re.DOTALL)
                            # if len(keys) > 1:
                            #     yield LLMStreamResponse(
                            #         error=True,
                            #         error_log="Parsing error : Nested key detected",
                            #     )
                            #     break
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

                if chunk.choices[0].finish_reason != None:
                    end_time = datetime.datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=functions,
                            function_call=function_call
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=tools,
                            tool_calls=tool_calls
                            if chunk.choices[0].finish_reason == "tool_calls"
                            else None,
                        )
                    )
        except Exception as e:
            logger.error(e)
            yield LLMStreamResponse(error=True, error_log=str(e))
