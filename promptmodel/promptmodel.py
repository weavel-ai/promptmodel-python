from __future__ import annotations
import asyncio

from dataclasses import dataclass
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

import promptmodel.utils.logger as logger
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.prompt_util import fetch_prompts, run_async_in_sync
from promptmodel.utils.types import LLMStreamResponse, LLMResponse
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

        # Get the current frame
        frame = sys._getframe(2)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of DevClient among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, DevClient):
                return var_val
        return None


class PromptModel(metaclass=RegisteringMeta):
    def __init__(self, name, rate_limit_manager=None):
        self.name = name
        self.llm_proxy = LLMProxy(name, rate_limit_manager)

    def get_prompts(self) -> List[Dict[str, str]]:
        """Get prompt for the promptmodel.
        If dev mode is running(if .promptmodel/config['dev_branch']['online'] = True), it will fetch the latest tested prompt in the dev branch local DB.
        If dev mode is not running, it will fetch the published prompt from the Cloud. (It will be saved in cache DB, so there is no extra latency for API call.)
        - If you made A/B testing in Web Dashboard, it will fetch the prompt randomly by the A/B testing ratio.
        If dev mode is initializing, it will return {}.

        Returns:
            List[Dict[str, str]]: list of prompts. Each prompt is a dict with 'role' and 'content'.
        """
        # add name to the list of prompt_models

        prompts, _ = run_async_in_sync(fetch_prompts(self.name))
        return prompts

    def run(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Run PromptModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return self.llm_proxy.run(inputs, function_list)

    async def arun(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Async run PromptModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return await self.llm_proxy.arun(inputs, function_list)

    def stream(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run PromptModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        for item in self.llm_proxy.stream(inputs, function_list):
            yield item

    async def astream(
        self,
        inputs: Optional[Dict[str, Any]] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Async Run PromptModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        async for item in self.llm_proxy.astream(inputs, function_list):
            yield item

    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Run PromptModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return self.llm_proxy.run_and_parse(inputs, function_list)

    async def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Async Run PromptModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        return await self.llm_proxy.arun_and_parse(inputs, function_list)

    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run PromptModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """
        for item in self.llm_proxy.stream_and_parse(inputs, function_list):
            yield item

    async def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        function_list: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Async Run PromptModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """
        async for item in self.llm_proxy.astream_and_parse(inputs, function_list):
            yield item

    # def run_and_parse_function_call(
    #     self,
    #     inputs: Dict[str, Any] = {},
    #     function_list: List[Callable[..., Any]] = [],
    # ) -> Generator[str, None, None]:
    #     return self.llm_proxy.run_and_parse_function_call(inputs, function_list)

    # async def arun_and_parse_function_call(
    #     self,
    #     inputs: Dict[str, Any] = {},
    #     function_list: List[Callable[..., Any]] = [],
    # ) -> Generator[str, None, None]:
    #     return await self.llm_proxy.arun_and_parse_function_call(inputs, function_list)

    # async def astream_and_parse_function_call(
    #     self,
    #     inputs: Dict[str, Any] = {},
    #     function_list: List[Callable[..., Any]] = [],
    # ) -> AsyncGenerator[str, None]:
    #     async for item in self.llm_proxy.astream_and_parse_function_call(
    #         inputs, function_list
    #     ):
    #         yield item
