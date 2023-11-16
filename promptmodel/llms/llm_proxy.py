import asyncio
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    Coroutine,
)
from threading import Thread
from concurrent.futures import Future
from rich import print
from litellm.utils import ModelResponse
from promptmodel.llms.llm import LLM
from promptmodel.utils import logger
from promptmodel.utils.prompt_util import (
    fetch_prompts,
    run_async_in_sync,
)
from promptmodel.utils.chat_util import (
    fetch_chat_model,
    fetch_chat_log,
)
from promptmodel.utils.output_utils import update_dict
from promptmodel.apis.base import APIClient, AsyncAPIClient
from promptmodel.utils.types import LLMResponse, LLMStreamResponse


class LLMProxy(LLM):
    def __init__(self, name: str, rate_limit_manager=None):
        super().__init__(rate_limit_manager)
        self._name = name

    def _wrap_gen(self, gen: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = run_async_in_sync(fetch_prompts(self._name))
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )
            # Call the generator with the arguments
            stream_response: Generator[LLMStreamResponse, None, None] = gen(**call_args)

            api_response = None
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            error_occurs = False
            error_log = None
            for item in stream_response:
                if (
                    item.api_response and "delta" not in item.api_response["choices"][0]
                ):  # only get the last api_response, not delta response
                    api_response = item.api_response
                if item.parsed_outputs:
                    dict_cache = update_dict(dict_cache, item.parsed_outputs)
                if item.raw_output:
                    string_cache += item.raw_output
                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log

                if error_occurs:
                    # delete all promptmodel data in item
                    item.raw_output = None
                    item.parsed_outputs = None
                    item.function_call = None
                yield item

            # add string_cache in model_response
            if api_response:
                if "message" not in api_response.choices[0]:
                    api_response.choices[0]["message"] = {}
                if "content" not in api_response.choices[0]["message"]:
                    api_response.choices[0]["message"]["content"] = string_cache
                    api_response.choices[0]["message"]["role"] = "assistant"

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            run_async_in_sync(
                self._async_log_to_cloud(
                    version_details["uuid"], inputs, api_response, dict_cache, metadata
                )
            )

        return wrapper

    def _wrap_async_gen(self, async_gen: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = await fetch_prompts(self._name)
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )

            # Call async_gen with the arguments
            stream_response: AsyncGenerator[LLMStreamResponse, None] = async_gen(
                **call_args
            )

            api_response = None
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            error_occurs = False
            error_log = None
            api_response: Optional[ModelResponse] = None
            async for item in stream_response:
                if (
                    item.api_response and "delta" not in item.api_response["choices"][0]
                ):  # only get the last api_response, not delta response
                    api_response = item.api_response
                if item.parsed_outputs:
                    dict_cache = update_dict(dict_cache, item.parsed_outputs)
                if item.raw_output:
                    string_cache += item.raw_output
                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log
                yield item

            # add string_cache in model_response
            if api_response:
                if "message" not in api_response.choices[0]:
                    api_response.choices[0]["message"] = {}
                if "content" not in api_response.choices[0]["message"]:
                    api_response.choices[0]["message"]["content"] = string_cache
                    api_response.choices[0]["message"]["role"] = "assistant"

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            await self._async_log_to_cloud(
                version_details["uuid"], inputs, api_response, dict_cache, metadata
            )

            # raise Exception("error_log")

        return wrapper

    def _wrap_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = run_async_in_sync(fetch_prompts(self._name))
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            if llm_response.parsed_outputs:
                run_async_in_sync(
                    self._async_log_to_cloud(
                        version_details["uuid"],
                        inputs,
                        llm_response.api_response,
                        llm_response.parsed_outputs,
                        metadata,
                    )
                )
            else:
                run_async_in_sync(
                    self._async_log_to_cloud(
                        version_details["uuid"],
                        inputs,
                        llm_response.api_response,
                        {},
                        metadata,
                    )
                )
            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None
            return llm_response

        return wrapper

    def _wrap_async_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = await fetch_prompts(
                self._name
            )  # messages, model, uuid = self._fetch_prompts()
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = await method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }

            if llm_response.parsed_outputs:
                await self._async_log_to_cloud(
                    version_details["uuid"],
                    inputs,
                    llm_response.api_response,
                    llm_response.parsed_outputs,
                    metadata,
                )
            else:
                await self._async_log_to_cloud(
                    version_details["uuid"],
                    inputs,
                    llm_response.api_response,
                    {},
                    metadata,
                )

            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None
            return llm_response

        return async_wrapper

    def _wrap_chat(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(session_uuid: str, **kwargs):
            message_logs = run_async_in_sync(fetch_chat_log(session_uuid))
            instruction, version_details = run_async_in_sync(
                fetch_chat_model(self._name, session_uuid)
            )

            if len(message_logs) == 0 or message_logs[0]["role"] != "system":
                message_logs = instruction + message_logs

            call_args = self._prepare_call_args_for_chat(
                message_logs, version_details, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            if llm_response.api_response:
                metadata["api_response"] = llm_response.api_response.to_dict_recursive()
                metadata["token_usage"] = llm_response.api_response["token_usage"]
                metadata["latency"] = llm_response.api_response["response_ms"]

            run_async_in_sync(
                self._async_chat_log_to_cloud(
                    session_uuid,
                    [llm_response.api_response["choices"][0]["message"]],
                    version_details["uuid"],
                    [metadata],
                )
            )

            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None

            return llm_response

        return wrapper

    def _wrap_async_chat(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(session_uuid: str, **kwargs):
            message_logs = await fetch_chat_log(session_uuid)
            instruction, version_details = await fetch_chat_model(
                self._name, session_uuid
            )

            if len(message_logs) == 0 or message_logs[0]["role"] != "system":
                message_logs = instruction + message_logs

            call_args = self._prepare_call_args_for_chat(
                message_logs, version_details, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = await method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            if llm_response.api_response:
                metadata["api_response"] = llm_response.api_response.to_dict_recursive()
                metadata["token_usage"] = llm_response.api_response["token_usage"]
                metadata["latency"] = llm_response.api_response["response_ms"]

            await self._async_chat_log_to_cloud(
                session_uuid,
                [llm_response.api_response["choices"][0]["message"]],
                version_details["uuid"],
                [metadata],
            )

            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None

            return llm_response

        return async_wrapper

    def _prepare_call_args(
        self,
        prompts: List[Dict[str, str]],
        version_detail: Dict[str, Any],
        inputs: Dict[str, Any],
        kwargs,
    ):
        stringified_inputs = {key: str(value) for key, value in inputs.items()}
        messages = [
            {
                "content": prompt["content"].format(**stringified_inputs),
                "role": prompt["role"],
            }
            for prompt in prompts
        ]
        call_args = {
            "messages": messages,
            "model": version_detail["model"] if version_detail else None,
            "parsing_type": version_detail["parsing_type"] if version_detail else None,
            "output_keys": version_detail["output_keys"] if version_detail else None,
        }
        if call_args["parsing_type"] is None:
            del call_args["parsing_type"]
            del call_args["output_keys"]

        if "function_list" in kwargs:
            call_args["functions"] = kwargs["function_list"]
        return call_args

    def _prepare_call_args_for_chat(
        self,
        messages: List[Dict[str, Any]],
        version_detail: Dict[str, Any],
        kwargs,
    ):
        # TODO: truncate messages to make length <= model's max length
        call_args = {
            "messages": messages,
            "model": version_detail["model"] if version_detail else None,
        }

        if "function_list" in kwargs:
            call_args["functions"] = kwargs["function_list"]
        return call_args

    async def _async_log_to_cloud(
        self,
        version_uuid: str,
        inputs: Optional[Dict] = None,
        api_response: Optional[ModelResponse] = None,
        parsed_outputs: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        # Perform the logging asynchronously
        if api_response:
            api_response_dict = api_response.to_dict_recursive()
            api_response_dict.update(
                {
                    "response_ms": api_response["response_ms"]
                    if "response_ms" in api_response
                    else api_response.response_ms
                }
            )
        else:
            api_response_dict = None
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/log_deployment_run",
            params={
                "version_uuid": version_uuid,
            },
            json={
                "inputs": inputs,
                "api_response": api_response_dict,
                "parsed_outputs": parsed_outputs,
                "metadata": metadata,
            },
            use_cli_key=False,
        )
        if res.status_code != 200:
            print(f"[red]Failed to log to cloud: {res.json()}[/red]")
        return res

    async def _async_chat_log_to_cloud(
        self,
        session_uuid: str,
        messages: List[Dict[str, Any]],
        version_uuid: Optional[str] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        # Perform the logging asynchronously
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/log_deployment_chat",
            params={
                "session_uuid": session_uuid,
                "version_uuid": version_uuid,
            },
            json={
                "messages": messages,
                "metadata": metadata,
            },
            use_cli_key=False,
        )
        if res.status_code != 200:
            print(f"[red]Failed to log to cloud: {res.json()}[/red]")
        return res

    def run(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> LLMResponse:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_method(super().run)(inputs, **kwargs)

    def arun(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> LLMResponse:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_async_method(super().arun)(inputs, **kwargs)

    def stream(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> Generator[LLMStreamResponse, None, None]:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_gen(super().stream)(inputs, **kwargs)

    def astream(
        self,
        inputs: Optional[Dict[str, Any]] = {},
        function_list: Optional[List[Any]] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_async_gen(super().astream)(inputs, **kwargs)

    def run_and_parse(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> LLMResponse:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_method(super().run_and_parse)(inputs, **kwargs)

    def arun_and_parse(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> LLMResponse:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_async_method(super().arun_and_parse)(inputs, **kwargs)

    def stream_and_parse(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> Generator[LLMStreamResponse, None, None]:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_gen(super().stream_and_parse)(inputs, **kwargs)

    def astream_and_parse(
        self, inputs: Dict[str, Any] = {}, function_list: Optional[List[Any]] = None
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_async_gen(super().astream_and_parse)(inputs, **kwargs)

    def chat_run(
        self,
        session_uuid: str,
        function_list: Optional[List[Any]] = None,
    ) -> LLMResponse:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_chat(super().run)(session_uuid, **kwargs)

    def chat_arun(
        self,
        session_uuid: str,
        function_list: Optional[List[Any]] = None,
    ) -> LLMResponse:
        kwargs = {"function_list": function_list} if function_list else {}
        return self._wrap_async_chat(super().arun)(session_uuid, **kwargs)

    # def chat_stream(
    #     self,
    #     session_uuid: str,
    #     function_list: Optional[List[Any]] = None,
    # ) -> LLMResponse:
    #     kwargs = {"function_list": function_list} if function_list else {}
    #     return self._wrap_chat_gen(super().stream)(session_uuid, **kwargs)

    # def chat_astream(
    #     self,
    #     session_uuid: str,
    #     function_list: Optional[List[Any]] = None,
    # ) -> LLMResponse:
    #     kwargs = {"function_list": function_list} if function_list else {}
    #     return self._wrap_async_chat_gen(super().astream)(session_uuid, **kwargs)
