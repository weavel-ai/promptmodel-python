from __future__ import annotations

from dataclasses import dataclass
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
from promptmodel.types.response import LLMStreamResponse, LLMResponse, PromptModelConfig
from promptmodel import DevClient


@dataclass
class PromptModelInterface:
    name: str
    default_model: str


class RegisteringMeta(type):
    def __call__(cls, *args, **kwargs):
        instance: "PromptModel" = super().__call__(*args, **kwargs)
        # Find the global client instance in the current context
        client: Optional[DevClient] = cls.find_client_instance()
        if client is not None:
            client.register_prompt_model(instance.name)
        return instance

    @staticmethod
    def find_client_instance():
        import sys

        # Get the current frame (frame where PromptModel is called)
        frame = sys._getframe(2)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of DevClient among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, DevClient):
                return var_val
        return None


class PromptModel(metaclass=RegisteringMeta):
    """

    Args:
        name (_type_): _description_
        version (Optional[ Union[str, int] ], optional): Choose which PromptModel version to use. Defaults to "deploy". It can be "deploy", "latest", or version number.
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

    @check_connection_status_decorator
    def get_config(self) -> PromptModelConfig:
        """Get config for the promptmodel.
        It will fetch the prompt and version you specified from the Cloud. (It will be saved in cache DB, so there is no extra latency for API call.)
        - If you made A/B testing in Web Dashboard, it will fetch the prompt randomly by the A/B testing ratio.
        If dev mode is initializing, it will return None

        Returns:
            PromptModelConfig: config for the promptmodel. It contains prompts and version_detail.
        """
        # add name to the list of prompt_models

        prompt, version_detail = run_async_in_sync(
            LLMProxy.fetch_prompts(self.name, self.version)
        )
        return PromptModelConfig(prompt, version_detail)

    @check_connection_status_decorator
    def run(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Run PromptModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return self.llm_proxy.run(inputs, functions, tools, self.api_key)

    @check_connection_status_decorator
    async def arun(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Async run PromptModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return await self.llm_proxy.arun(inputs, functions, tools, self.api_key)

    @check_connection_status_decorator
    def stream(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run PromptModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        for item in self.llm_proxy.stream(inputs, functions, tools, self.api_key):
            yield item

    @check_connection_status_decorator
    async def astream(
        self,
        inputs: Optional[Dict[str, Any]] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Coroutine[AsyncGenerator[LLMStreamResponse, None]]:
        """Async Run PromptModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return (
            item
            async for item in self.llm_proxy.astream(
                inputs, functions, tools, self.api_key
            )
        )

    @check_connection_status_decorator
    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Run PromptModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return self.llm_proxy.run_and_parse(inputs, functions, tools, self.api_key)

    @check_connection_status_decorator
    async def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Async Run PromptModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return await self.llm_proxy.arun_and_parse(
            inputs, functions, tools, self.api_key
        )

    @check_connection_status_decorator
    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run PromptModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """
        for item in self.llm_proxy.stream_and_parse(
            inputs, functions, tools, self.api_key
        ):
            yield item

    @check_connection_status_decorator
    async def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Coroutine[AsyncGenerator[LLMStreamResponse, None]]:
        """Async Run PromptModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """
        return (
            item
            async for item in self.llm_proxy.astream_and_parse(
                inputs, functions, tools, self.api_key
            )
        )
