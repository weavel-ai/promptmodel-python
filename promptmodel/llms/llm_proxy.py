import inspect
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
from litellm import RateLimitManager, ModelResponse
from promptmodel.llms.llm import LLM
from promptmodel.utils import logger
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils.prompt_util import fetch_prompts
from promptmodel.apis.base import APIClient, AsyncAPIClient

class LLMProxy(LLM):
    def __init__(
        self, name: str, rate_limit_manager: Optional[RateLimitManager] = None
    ):
        super().__init__(rate_limit_manager)
        self._name = name

    def _wrap_gen(self, gen: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = asyncio.run(fetch_prompts(self._name))
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            call_args = self._prepare_call_args(prompts, version_details['model'], inputs, True, kwargs)
            # Call the generator with the arguments
            gen_instance = gen(**call_args)

            raw_response = None
            for item in gen_instance:
                if isinstance(item, ModelResponse):
                    raw_response = item
                elif isinstance(item, dict):
                    for key, value in item.items():
                        dict_cache[key] = dict_cache.get(key, "") + value
                elif isinstance(item, str):
                    string_cache += item
                yield item

            self._log_to_cloud(version_details['uuid'], inputs, raw_response, dict_cache)

        return wrapper

    def _wrap_async_gen(self, async_gen: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = await fetch_prompts(self._name)
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            call_args = self._prepare_call_args(prompts, version_details['model'], inputs, True, kwargs)

            # Call async_gen with the arguments
            async_gen_instance = async_gen(**call_args)

            raw_response = None
            async for item in async_gen_instance:
                if isinstance(item, ModelResponse):
                    raw_response = item
                elif isinstance(item, dict):
                    for key, value in item.items():
                        dict_cache[key] = dict_cache.get(key, "") + value
                elif isinstance(item, str):
                    string_cache += item
                yield item

            self._log_to_cloud(version_details['uuid'], inputs, raw_response, dict_cache)
            
        return wrapper

    def _wrap_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = asyncio.run(fetch_prompts(self._name))
            call_args = self._prepare_call_args(prompts, version_details['model'], inputs, True, kwargs)
            
            # Call the method with the arguments
            result, raw_response = method(**call_args)
            if isinstance(result, dict):
                self._log_to_cloud(version_details['uuid'], inputs, raw_response, result)
            else:
                self._log_to_cloud(version_details['uuid'], inputs, raw_response, {})
            return result

        return wrapper

    def _wrap_async_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = await fetch_prompts(self._name) # messages, model, uuid = self._fetch_prompts()
            call_args = self._prepare_call_args(prompts, version_details['model'], inputs, True, kwargs)

            # Call the method with the arguments
            result, raw_response = await method(**call_args)
            if isinstance(result, dict):
                self._log_to_cloud(version_details['uuid'], inputs, raw_response, result)
            else:
                self._log_to_cloud(version_details['uuid'], inputs, raw_response, {})
            return result 
        return async_wrapper

    def _prepare_call_args(
        self,
        prompts: List[Dict[str, str]],
        model: str,
        inputs: Dict[str, Any],
        show_response: bool,
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
            "model": model,
            "show_response": show_response,
        }

        if "function_list" in kwargs:
            call_args["function_list"] = kwargs["function_list"]

        return call_args

    def _log_to_cloud(
        self,
        version_uuid: str,
        inputs: dict,
        raw_response: ModelResponse,
        parsed_outputs: dict,
    ):
        # Log to cloud
        # logging if only status = deployed
        config = read_config()
        if "dev_branch" in config and (
            config["dev_branch"]["initializing"] or config["dev_branch"]["online"]
        ):
            return

        raw_response_dict = raw_response.to_dict_recursive()
        raw_response_dict.update({"response_ms": raw_response.response_ms})
        res = asyncio.run(
            AsyncAPIClient.execute(
                method="POST",
                path="/log_deployment_run",
                params={
                    "version_uuid": version_uuid,
                },
                json={
                    "inputs": inputs,
                    "raw_response": raw_response_dict,
                    "parsed_outputs": parsed_outputs,
                },
                use_cli_key=False,
            )
        )
        if res.status_code != 200:
            print(f"[red]Failed to log to cloud: {res.json()}[/red]")
        return

    def run(self, inputs: Dict[str, Any] = {}) -> str:
        return self._wrap_method(super().run)(inputs)

    def arun(self, inputs: Dict[str, Any] = {}) -> str:
        return self._wrap_async_method(super().arun)(inputs)

    # def run_function_call(self, inputs: Dict[str, Any] = {}) -> Tuple[Any, Any]:
    #     return self._wrap_method(super().run_function_call)(inputs)

    # def arun_function_call(self, inputs: Dict[str, Any] = {}) -> Tuple[Any, Any]:
    #     return self._wrap_async_method(super().arun_function_call)(inputs)

    def stream(self, inputs: Dict[str, Any] = {}) -> Generator[str, None, None]:
        return self._wrap_gen(super().stream)(inputs)

    def astream(
        self, inputs: Optional[Dict[str, Any]] = {}
    ) -> AsyncGenerator[str, None]:
        return self._wrap_async_gen(super().astream)(inputs)

    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
    ) -> Dict[str, str]:
        return self._wrap_method(super().run_and_parse)(inputs)

    def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
    ) -> Dict[str, str]:
        return self._wrap_async_gen(super().arun_and_parse)(inputs)

    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
    ) -> Generator[str, None, None]:
        return self._wrap_gen(super().stream_and_parse)(inputs)

    def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
    ) -> AsyncGenerator[str, None]:
        return self._wrap_async_gen(super().astream_and_parse)(inputs)

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
