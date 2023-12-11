from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Coroutine,
    Union,
)

import promptmodel.utils.logger as logger
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.async_utils import run_async_in_sync
from promptmodel.utils.config_utils import check_connection_status_decorator
from promptmodel.types.response import (
    LLMStreamResponse,
    LLMResponse,
    FunctionModelConfig,
)
from promptmodel.types.enums import InstanceType
from promptmodel.apis.base import AsyncAPIClient
from promptmodel import DevClient


@dataclass
class FunctionModelInterface:
    name: str
    default_model: str


class RegisteringMeta(type):
    def __call__(cls, *args, **kwargs):
        instance: "FunctionModel" = super().__call__(*args, **kwargs)
        # Find the global client instance in the current context
        client: Optional[DevClient] = cls.find_client_instance()
        if client is not None:
            client.register_function_model(instance.name)
        return instance

    @staticmethod
    def find_client_instance():
        import sys

        # Get the current frame (frame where FunctionModel is called)
        frame = sys._getframe(2)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of DevClient among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, DevClient):
                return var_val
        return None


class FunctionModel(metaclass=RegisteringMeta):
    """

    Args:
        name (_type_): _description_
        version (Optional[ Union[str, int] ], optional): Choose which FunctionModel version to use. Defaults to "deploy". It can be "deploy", "latest", or version number.
        api_key (Optional[str], optional): API key for the LLM. Defaults to None. If None, use api_key in .env file.
    """

    def __init__(
        self,
        name,
        version: Optional[
            Union[str, int]
        ] = "deploy",  # "deploy" or "latest" or version number
        api_key: Optional[str] = None,
    ):
        self.name = name
        self.api_key = api_key
        self.llm_proxy = LLMProxy(name, version)
        self.version = version
        self.recent_log_uuid = None

    @check_connection_status_decorator
    def get_config(self, *args, **kwargs) -> FunctionModelConfig:
        """Get config for the promptmodel.
        It will fetch the prompt and version you specified from the Cloud. (It will be saved in cache DB, so there is no extra latency for API call.)
        - If you made A/B testing in Web Dashboard, it will fetch the prompt randomly by the A/B testing ratio.
        If dev mode is initializing, it will return None

        Returns:
            FunctionModelConfig: config for the promptmodel. It contains prompts and version_detail.
        """
        # add name to the list of function_models

        prompt, version_detail = run_async_in_sync(
            LLMProxy.fetch_prompts(self.name, self.version)
        )
        return FunctionModelConfig(prompt, version_detail)

    @check_connection_status_decorator
    def run(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Run FunctionModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = self.llm_proxy.run(inputs, functions, tools, self.api_key)
        self.recent_log_uuid = res.pm_log_uuid
        return res

    @check_connection_status_decorator
    async def arun(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Async run FunctionModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = await self.llm_proxy.arun(
            inputs, functions, tools, self.api_key
        )
        self.recent_log_uuid = res.pm_log_uuid
        return res

    @check_connection_status_decorator
    def stream(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run FunctionModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        cache: Optional[LLMStreamResponse] = None
        for item in self.llm_proxy.stream(inputs, functions, tools, self.api_key):
            yield item
            cache = item

        if cache:
            self.recent_log_uuid = cache.pm_log_uuid

    @check_connection_status_decorator
    async def astream(
        self,
        inputs: Optional[Dict[str, Any]] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Coroutine[AsyncGenerator[LLMStreamResponse, None]]:
        """Async Run FunctionModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """

        async def async_gen():
            cache: Optional[LLMStreamResponse] = None
            async for item in self.llm_proxy.astream(
                inputs, functions, tools, self.api_key
            ):
                yield item
                cache = item
            if cache:
                self.recent_log_uuid = cache.pm_log_uuid

        return async_gen()

    @check_connection_status_decorator
    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Run FunctionModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = self.llm_proxy.run_and_parse(
            inputs, functions, tools, self.api_key
        )
        self.recent_log_uuid = res.pm_log_uuid
        return res

    @check_connection_status_decorator
    async def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Async Run FunctionModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = await self.llm_proxy.arun_and_parse(
            inputs, functions, tools, self.api_key
        )
        self.recent_log_uuid = res.pm_log_uuid
        return res

    @check_connection_status_decorator
    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run FunctionModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """
        cache: Optional[LLMStreamResponse] = None
        for item in self.llm_proxy.stream_and_parse(
            inputs, functions, tools, self.api_key
        ):
            yield item
            cache = item

        if cache:
            self.recent_log_uuid = cache.pm_log_uuid

    @check_connection_status_decorator
    async def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Coroutine[AsyncGenerator[LLMStreamResponse, None]]:
        """Async Run FunctionModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """

        async def async_gen():
            cache: Optional[LLMStreamResponse] = None
            async for item in self.llm_proxy.astream_and_parse(
                inputs, functions, tools, self.api_key
            ):
                yield item
                cache = item
            if cache:
                self.recent_log_uuid = cache.pm_log_uuid

        return async_gen()

    @check_connection_status_decorator
    async def log(
        self,
        log_uuid: Optional[str] = None,
        content: Optional[Dict[str, Any]] = {},  # TODO: FIX THIS INTO OPENAI OUTPUT
        metadata: Optional[Dict[str, Any]] = {},
        *args,
        **kwargs,
    ):
        try:
            if not log_uuid and self.recent_log_uuid is not None:
                log_uuid = self.recent_log_uuid
            config = kwargs["config"]
            if (
                "mask_inputs" in config["project"]
                and config["project"]["mask_inputs"] is True
            ):
                if "inputs" in content:
                    content["inputs"] = {
                        key: "PRIVATE LOGGING"
                        for key, value in content["inputs"].items()
                    }
            res = await AsyncAPIClient.execute(
                method="POST",
                path="/log_general",
                params={"type": InstanceType.RunLog.value, "identifier": log_uuid},
                json={"content": content, "metadata": metadata},
                use_cli_key=False,
            )
            if res.status_code != 200:
                logger.error(f"Logging error: {res}")
        except Exception as exception:
            logger.error(f"Logging error: {exception}")


class PromptModel(FunctionModel):
    """Deprecated"""

    def __init__(self, name, version: Optional[Union[str, int]] = "deploy"):
        super().__init__(name, version)
