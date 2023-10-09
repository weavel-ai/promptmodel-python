import inspect
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
from .llm import LLM


class LLMProxy(LLM):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def _wrap_gen(self, gen: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            messages, model = self._fetch_from_cloud()
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            # TODO: Fill in the messages with inputs

            # Check for specific keys in kwargs
            call_args = {"messages": messages, "model": model}

            if "output_keys" in kwargs:
                call_args["output_keys"] = kwargs["output_keys"]
            if "function_list" in kwargs:
                call_args["function_list"] = kwargs["function_list"]

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
            messages, model = self._fetch_from_cloud()
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            # TODO: Fill in the messages with inputs

            call_args = {"messages": messages, "model": model}

            if "output_keys" in kwargs:
                call_args["output_keys"] = kwargs["output_keys"]
            if "function_list" in kwargs:
                call_args["function_list"] = kwargs["function_list"]

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
            messages, model = self._fetch_from_cloud()
            # TODO: Fill in the messages with inputs

            # Check for specific keys in kwargs
            call_args = {"messages": messages, "model": model}

            if "output_keys" in kwargs:
                call_args["output_keys"] = kwargs["output_keys"]
            if "function_list" in kwargs:
                call_args["function_list"] = kwargs["function_list"]

            # Call the method with the arguments
            result = method(**call_args)
            self._log_to_cloud(result)
            return result

        return wrapper

    def _wrap_async_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(inputs: Dict[str, Any], **kwargs):
            messages, model = self._fetch_from_cloud()
            # TODO: Fill in the messages with inputs

            # Check for specific keys in kwargs
            call_args = {"messages": messages, "model": model}

            if "output_keys" in kwargs:
                call_args["output_keys"] = kwargs["output_keys"]
            if "function_list" in kwargs:
                call_args["function_list"] = kwargs["function_list"]

            # Call the method with the arguments
            result = await method(**call_args)
            self._log_to_cloud(result)
            return result

        return async_wrapper

    def _fetch_from_cloud(self) -> Tuple[List[Dict[str, str]], str]:
        # TODO: Fetch from cloud
        return [
            {"role": "system", "content": "Help the user."},
            {"role": "user", "content": "What's your name?"},
        ], "gpt-3.5-turbo"

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
