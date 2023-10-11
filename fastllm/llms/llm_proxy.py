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
from fastllm.llms.llm import LLM
from fastllm.utils.config_utils import read_config, upsert_config
from fastllm.utils.prompt_util import fetch_prompts
from fastllm.apis.base import APIClient
from fastllm.database.crud import (
    get_latest_version_prompts,
    get_deployed_prompts,
    update_deployed_cache
)
class LLMProxy(LLM):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def _wrap_gen(self, gen: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, model = asyncio.run(fetch_prompts(self._name))
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            call_args = self._prepare_call_args(prompts, model, inputs, kwargs)
            # Call the generator with the arguments
            gen_instance = gen(**call_args)

            for item in gen_instance:
                if isinstance(item, dict):
                    for key, value in item.items():
                        dict_cache[key] = dict_cache.get(key, "") + value
                elif isinstance(item, str):
                    string_cache += item
                yield item

            if dict_cache:
                self._log_to_cloud(dict_cache)  # log the aggregated dictionary cache
            if string_cache:
                self._log_to_cloud(string_cache)  # log the aggregated string cache

        return wrapper

    def _wrap_async_gen(self, async_gen: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, model = await fetch_prompts(self._name)
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            call_args = self._prepare_call_args(prompts, model, inputs, kwargs)

            # Call async_gen with the arguments
            async_gen_instance = async_gen(**call_args)

            async for item in async_gen_instance:
                if isinstance(item, dict):
                    for key, value in item.items():
                        dict_cache[key] = dict_cache.get(key, "") + value
                elif isinstance(item, str):
                    string_cache += item
                yield item

            if dict_cache:
                self._log_to_cloud(dict_cache)  # log the aggregated dictionary cache
            if string_cache:
                self._log_to_cloud(string_cache)  # log the aggregated string cache

        return wrapper

    def _wrap_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, model = asyncio.run(fetch_prompts(self._name))
            call_args = self._prepare_call_args(prompts, model, inputs, kwargs)

            # Call the method with the arguments
            result = method(**call_args)
            self._log_to_cloud(result)
            return result

        return wrapper

    def _wrap_async_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, model = await fetch_prompts(self._name) # messages, model, prompt_metadata = self._fetch_prompts()
            call_args = self._prepare_call_args(prompts, model, inputs, kwargs)

            # Call the method with the arguments
            result = await method(**call_args)
            self._log_to_cloud(result)
            return result # return FastLLMOutput = {result, prompt_metadata, output_metadata (token_usage, cost, latency)}

        return async_wrapper

    def _prepare_call_args(self, prompts, model, inputs, kwargs):
        messages = [{'content': prompt['content'].format(**inputs), 'role': prompt['role']} for prompt in prompts]
        call_args = {"messages": messages, "model": model}

        if "output_keys" in kwargs:
            call_args["output_keys"] = kwargs["output_keys"]
        if "function_list" in kwargs:
            call_args["function_list"] = kwargs["function_list"]
            
        return call_args

    def _log_to_cloud(self, output: Union[str, Dict[str, str]]):
        # TODO: Log to cloud
        print(f"Logging to cloud: {output}")

    def generate(self, inputs: Dict[str, Any] = {}) -> str:
        return self._wrap_method(super().generate)(inputs)

    def agenerate(self, inputs: Dict[str, Any] = {}) -> str:
        return self._wrap_async_method(super().agenerate)(inputs)

    def stream(self, inputs: Dict[str, Any] = {}) -> Generator[str, None, None]:
        return self._wrap_gen(super().stream)(inputs)

    def astream(
        self, inputs: Optional[Dict[str, Any]] = {}
    ) -> AsyncGenerator[str, None]:
        return self._wrap_async_gen(super().astream)(inputs)

    def generate_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Dict[str, str]:
        return self._wrap_gen(super().generate_and_parse)(inputs, output_keys)

    def agenerate_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Dict[str, str]:
        return self._wrap_async_gen(super().agenerate_and_parse)(inputs, output_keys)

    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> Generator[str, None, None]:
        return self._wrap_gen(super().stream_and_parse)(inputs, output_keys)

    def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        output_keys: List[str] = [],
    ) -> AsyncGenerator[str, None]:
        return self._wrap_async_gen(super().astream_and_parse)(inputs, output_keys)

    def generate_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> Generator[str, None, None]:
        return self._wrap_method(super().generate_and_parse_function_call)(
            inputs, function_list
        )

    def agenerate_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> Generator[str, None, None]:
        return self._wrap_async_method(super().agenerate_and_parse_function_call)(
            inputs, function_list
        )

    def astream_and_parse_function_call(
        self,
        inputs: Dict[str, Any] = {},
        function_list: List[Callable[..., Any]] = [],
    ) -> AsyncGenerator[str, None]:
        return self._wrap_async_gen(super().astream_and_parse_function_call)(
            inputs, function_list
        )
        