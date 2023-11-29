from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Coroutine, Union
from uuid import uuid4
from promptmodel import DevClient

from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils import logger
from promptmodel.utils.config_utils import (
    read_config,
    upsert_config,
    check_connection_status_decorator,
)
from promptmodel.utils.async_utils import run_async_in_sync
from promptmodel.types.response import LLMStreamResponse, LLMResponse, ChatModelConfig


class RegisteringMeta(type):
    def __call__(cls, *args, **kwargs):
        instance: ChatModel = super().__call__(*args, **kwargs)
        # Find the global client instance in the current context
        client = cls.find_client_instance()
        if client is not None:
            client.register_chat_model(instance.name)
        return instance

    @staticmethod
    def find_client_instance():
        import sys

        # Get the current frame
        frame = sys._getframe(2)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of Client among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, DevClient):
                return var_val
        return None


class ChatModel(metaclass=RegisteringMeta):
    """

    Args:
        name (_type_): _description_
        version (Optional[ Union[str, int] ], optional): Choose which PromptModel version to use. Defaults to "deploy". It can be "deploy", "latest", or version number.
        api_key (Optional[str], optional): API key for the LLM. Defaults to None. If None, use api_key in .env file.
    """

    def __init__(
        self,
        name,
        session_uuid: str = None,
        version: Optional[Union[str, int]] = "deploy",
        api_key: Optional[str] = None,
    ):
        self.name = name
        self.api_key = api_key
        self.llm_proxy = LLMProxy(name, version)
        self.version = version

        if session_uuid is None:
            self.session_uuid = uuid4()
            instruction, version_details, chat_logs = run_async_in_sync(
                LLMProxy.fetch_chat_model(self.name, None, version)
            )
            config = read_config()
            if (
                "connection" in config
                and "initializing" in config["connection"]
                and config["connection"]["initializing"] == True
            ):
                return
            elif (
                "connection" in config
                and "reloading" in config["connection"]
                and config["connection"]["reloading"] == True
            ):
                return
            else:
                run_async_in_sync(
                    self.llm_proxy._async_make_session_cloud(
                        self.session_uuid,
                        version_details["uuid"],
                    )
                )
        else:
            self.session_uuid = session_uuid

    def get_config(self) -> ChatModelConfig:
        """Get config for the ChatModel.
        It will fetch the published prompt and version config from the Cloud. (It will be saved in cache DB, so there is no extra latency for API call.)
        - If you made A/B testing in Web Dashboard, it will fetch the prompt randomly by the A/B testing ratio.
        If dev mode is initializing, it will return None

        Returns:
            ChatModelConfig: config for the ChatModel, which contains prompts and version_detail, message_logs
        """
        prompt, version_detail, message_logs = run_async_in_sync(
            LLMProxy.fetch_chat_model(self.name, self.session_uuid, self.version)
        )
        return ChatModelConfig(prompt, version_detail, message_logs)

    @check_connection_status_decorator
    def add_messages(
        self,
        new_messages: List[Dict[str, Any]],
        metadata_list: List[Optional[Dict]] = [],
    ) -> None:
        """Add messages to the chat model.

        Args:
            new_messages (List[Dict[str, Any]]): list of messages. Each message is a dict with 'role', 'content', and 'function_call'.
        """
        # Save messages to Cloud DB

        run_async_in_sync(
            self.llm_proxy._async_chat_log_to_cloud(
                self.session_uuid,
                new_messages,
                None,
                [{} for _ in range(len(new_messages))],
            )
        )

    @check_connection_status_decorator
    def run(
        self,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = False,
    ) -> LLMResponse:
        """Run PromptModel. It does not raise error.

        Args:
            functions (List[Dict[str, Any]], optional): list of functions to run. Defaults to None.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        if stream:

            def gen():
                for item in self.llm_proxy.chat_stream(
                    self.session_uuid, functions, tools
                ):
                    yield item

            return gen()
        else:
            return self.llm_proxy.chat_run(self.session_uuid, functions, tools)
        # return self.llm_proxy.chat_run(self.session_uuid, functions, self.api_key)

    @check_connection_status_decorator
    async def arun(
        self,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = False,
    ) -> LLMResponse:
        """Async run PromptModel. It does not raise error.

        Args:
            functions (List[Dict[str, Any]], optional): list of functions to run. Defaults to None.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        if stream:

            async def async_gen():
                async for item in self.llm_proxy.chat_astream(
                    self.session_uuid, functions, tools
                ):
                    yield item

            return async_gen()
        else:
            return await self.llm_proxy.chat_arun(self.session_uuid, functions)
        # return await self.llm_proxy.chat_arun(
        #     self.session_uuid, functions, self.api_key
        # )
