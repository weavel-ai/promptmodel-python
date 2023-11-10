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
)
from rich import print
from litellm.utils import ModelResponse
from promptmodel.llms.llm import LLM
from promptmodel.utils import logger
from promptmodel.utils.prompt_util import fetch_prompts, run_async_in_sync
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

            raw_response = None
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            error_occurs = False
            error_log = None
            for item in stream_response:
                if item.api_response:
                    raw_response = item.api_response
                if item.parsed_outputs:
                    dict_cache = update_dict(dict_cache, item.parsed_outputs)
                if item.raw_output:
                    string_cache += item.raw_output
                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log
                yield item

            # add string_cache in model_response
            if raw_response:
                if "message" not in raw_response.choices[0]:
                    raw_response.choices[0]["message"] = {}
                if "content" not in raw_response.choices[0]["message"]:
                    raw_response.choices[0]["message"]["content"] = string_cache
                    raw_response.choices[0]["message"]["role"] = "assistant"

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            self._sync_log_to_cloud(
                version_details["uuid"], inputs, raw_response, dict_cache, metadata
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

            raw_response = None
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            error_occurs = False
            error_log = None
            raw_response: Optional[ModelResponse] = None
            async for item in stream_response:
                if item.api_response:
                    raw_response = item.api_response
                if item.parsed_outputs:
                    dict_cache = update_dict(dict_cache, item.parsed_outputs)
                if item.raw_output:
                    string_cache += item.raw_output
                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log
                yield item

            # add string_cache in model_response
            if raw_response:
                if "message" not in raw_response.choices[0]:
                    raw_response.choices[0]["message"] = {}
                if "content" not in raw_response.choices[0]["message"]:
                    raw_response.choices[0]["message"]["content"] = string_cache
                    raw_response.choices[0]["message"]["role"] = "assistant"

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            await self._async_log_to_cloud(
                version_details["uuid"], inputs, raw_response, dict_cache, metadata
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
                self._sync_log_to_cloud(
                    version_details["uuid"],
                    inputs,
                    llm_response.api_response,
                    llm_response.parsed_outputs,
                    metadata,
                )
            else:
                self._sync_log_to_cloud(
                    version_details["uuid"],
                    inputs,
                    llm_response.api_response,
                    {},
                    metadata,
                )
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

    def _sync_log_to_cloud(
        self,
        version_uuid: str,
        inputs: Optional[Dict] = None,
        api_response: Optional[ModelResponse] = None,
        parsed_outputs: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        coro = self._async_log_to_cloud(
            version_uuid,
            inputs,
            api_response,
            parsed_outputs,
            metadata,
        )
        # This is the synchronous wrapper that will wait for the async function to complete.
        try:
            # Try to get a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If there is no running loop, create a new one and set it as the event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            # loop.close()  # Make sure to close the loop after use
            return result

        if loop.is_running():
            # nest_asyncio.apply already done
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)

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

    # def run_and_parse_function_call(
    #     self,
    #     inputs: Dict[str, Any] = {},
    #     function_list: List[Callable[..., Any]] = [],
    # ) -> Generator[str, None, None]:
    #     return self._wrap_method(super().run_and_parse_function_call)(
    #         inputs, function_list
    #     )

    # def arun_and_parse_function_call(
    #     self,
    #     inputs: Dict[str, Any] = {},
    #     function_list: List[Callable[..., Any]] = [],
    # ) -> Generator[str, None, None]:
    #     return self._wrap_async_method(super().arun_and_parse_function_call)(
    #         inputs, function_list
    #     )

    # def astream_and_parse_function_call(
    #     self,
    #     inputs: Dict[str, Any] = {},
    #     function_list: List[Callable[..., Any]] = [],
    # ) -> AsyncGenerator[str, None]:
    #     return self._wrap_async_gen(super().astream_and_parse_function_call)(
    #         inputs, function_list
    #     )
