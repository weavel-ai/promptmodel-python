"""LLM for Development TestRun"""
import re
from datetime import datetime
from typing import Any, AsyncGenerator, List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import acompletion, get_max_tokens

from promptmodel.types.enums import ParsingType, get_pattern_by_type
from promptmodel.utils import logger
from promptmodel.utils.output_utils import convert_str_to_type, update_dict
from promptmodel.utils.token_counting import (
    num_tokens_for_messages,
    num_tokens_for_messages_for_each,
    num_tokens_from_functions_input,
)
from promptmodel.types.response import (
    LLMResponse,
    LLMStreamResponse,
    ModelResponse,
    Usage,
    Choices,
    Message,
)

load_dotenv()

class OpenAIMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = ""
    function_call: Optional[Dict[str, Any]] = None
    name: Optional[str] = None


class LLMDev:
    def __init__(self):
        self._model: str

    def __validate_openai_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        res = []
        for message in messages:
            res.append(OpenAIMessage(**message))
        return res

    async def dev_run(
        self,
        messages: List[Dict[str, Any]],
        parsing_type: Optional[ParsingType] = None,
        functions: Optional[List[Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        if functions == []:
            functions = None
            
        start_time = datetime.now()
        
        response: AsyncGenerator[ModelResponse, None] = await acompletion(
            model=_model,
            messages=[
                message.model_dump(exclude_none=True)
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
            functions=functions,
            **kwargs,
        )
        function_call = {"name": "", "arguments": ""}
        finish_reason_function_call = False
        async for chunk in response:
            if getattr(chunk.choices[0].delta, "content", None) is not None:
                stream_value = chunk.choices[0].delta.content
                raw_output += stream_value  # append raw output
                yield LLMStreamResponse(raw_output=stream_value)  # return raw output

            if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                for key, value in (
                    chunk.choices[0].delta.function_call.model_dump().items()
                ):
                    if value is not None:
                        function_call[key] += value

            if chunk.choices[0].finish_reason == "function_call":
                finish_reason_function_call = True
                yield LLMStreamResponse(function_call=function_call)
                
            if chunk.choices[0].finish_reason != None:
                end_time = datetime.now()
                response_ms = (end_time - start_time).total_seconds() * 1000
                yield LLMStreamResponse(
                    api_response=self.make_model_response_dev(
                        chunk,
                        response_ms,
                        messages,
                        raw_output,
                        functions=functions,
                        function_call=function_call
                        if chunk.choices[0].finish_reason == "function_call"
                        else None,
                        tools=None,
                        tool_calls=None
                    )
                )

        # parsing
        if parsing_type and not finish_reason_function_call:
            parsing_pattern: Dict[str, str] = get_pattern_by_type(parsing_type)
            whole_pattern = parsing_pattern["whole"]
            parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
            for parsed_result in parsed_results:
                key = parsed_result[0]
                type_str = parsed_result[1]
                value = convert_str_to_type(parsed_result[2], type_str)
                yield LLMStreamResponse(parsed_outputs={key: value})

    async def dev_chat(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        if functions == []:
            functions = None

        if model != "HCX-002":
            # Truncate the output if it is too long
            # truncate messages to make length <= model's max length
            token_per_functions = num_tokens_from_functions_input(
                functions=functions, model=model
            )
            model_max_tokens = get_max_tokens(model=model)
            token_per_messages = num_tokens_for_messages_for_each(messages, model)
            token_limit_exceeded = (
                sum(token_per_messages) + token_per_functions
            ) - model_max_tokens
            if token_limit_exceeded > 0:
                while token_limit_exceeded > 0:
                    # erase the second oldest message (first one is system prompt, so it should not be erased)
                    if len(messages) == 1:
                        # if there is only one message, Error cannot be solved. Just call LLM and get error response
                        break
                    token_limit_exceeded -= token_per_messages[1]
                    del messages[1]
                    del token_per_messages[1]

        args = dict(
            model=_model,
            messages=[
                message.model_dump(exclude_none=True)
                for message in self.__validate_openai_messages(messages)
            ],
            functions=functions,
            tools=tools,
        )

        is_stream_unsupported = model in ["HCX-002"]
        if not is_stream_unsupported:
            args["stream"] = True
        
        start_time = datetime.now()
        response: AsyncGenerator[ModelResponse, None] = await acompletion(**args, **kwargs)
        if is_stream_unsupported:
            yield LLMStreamResponse(raw_output=response.choices[0].message.content)
        else:
            async for chunk in response:
                yield_api_response_with_fc = False
                logger.debug(chunk)
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
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
                    
                if getattr(chunk.choices[0].delta, "finish_reason", None) is not None:
                    end_time = datetime.now()
                    response_ms = (end_time - start_time).total_seconds() * 1000
                    yield LLMStreamResponse(
                        api_response=self.make_model_response_dev(
                            chunk,
                            response_ms,
                            messages,
                            raw_output,
                            functions=None,
                            function_call=None
                            if chunk.choices[0].finish_reason == "function_call"
                            else None,
                            tools=None,
                            tool_calls=None
                        )
                    )

    def make_model_response_dev(
        self,
        chunk: ModelResponse,
        response_ms,
        messages: List[Dict[str, str]],
        raw_output: str,
        functions: Optional[List[Any]] = None,
        function_call: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Make ModelResponse object from openai response."""
        count_start_time = datetime.now()
        prompt_token: int = num_tokens_for_messages(
            messages=messages, model=chunk["model"]
        )
        completion_token: int = num_tokens_for_messages(
            model=chunk["model"],
            messages=[{"role": "assistant", "content": raw_output}],
        )

        if functions and len(functions) > 0:
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

        count_end_time = datetime.now()
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