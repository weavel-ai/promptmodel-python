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
    Coroutine,
)
from uuid import uuid4
from promptmodel import DevClient

from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.database.crud import create_chat_logs
from promptmodel.utils import logger
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils.prompt_util import (
    run_async_in_sync,
)
from promptmodel.utils.chat_util import (
    fetch_chat_model,
    fetch_chat_log,
)
from promptmodel.utils.types import LLMStreamResponse, LLMResponse


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
    def __init__(self, name, rate_limit_manager=None, chat_uuid: str = None):
        self.name = name
        self.llm_proxy = LLMProxy(name, rate_limit_manager)
        if chat_uuid is None:
            self.chat_uuid = uuid4()
            instruction, version_details = run_async_in_sync(
                fetch_chat_model(self.name)
            )
            config = read_config()
            if "dev_branch" in config and config["dev_branch"]["initializing"] == True:
                pass
            elif "dev_branch" in config and config["dev_branch"]["online"] == True:
                # if dev online=True, save in Local DB
                create_chat_logs(self.chat_uuid, instruction, version_details["uuid"])
            else:
                run_async_in_sync(
                    self.llm_proxy._async_chat_log_to_cloud(
                        self.chat_uuid,
                        version_details["uuid"],
                        instruction,
                        {},
                    )
                )
        else:
            self.chat_uuid = chat_uuid

    def get_prompts(self) -> List[Dict[str, str]]:
        """Get prompt for the promptmodel.
        If dev mode is running(if .promptmodel/config['dev_branch']['online'] = True), it will fetch the latest tested prompt in the dev branch local DB.
        If dev mode is not running, it will fetch the published prompt from the Cloud. (It will be saved in cache DB, so there is no extra latency for API call.)
        - If you made A/B testing in Web Dashboard, it will fetch the prompt randomly by the A/B testing ratio.
        If dev mode is initializing, it will return {}.

        Returns:
            List[Dict[str, str]]: list of prompts. Each prompt is a dict with 'role' and 'content'.
        """
        instruction, detail = run_async_in_sync(
            fetch_chat_model(self.name, self.chat_uuid)
        )
        return instruction

    def add_messages(self, new_messages: List[Dict[str, Any]]) -> None:
        """Add messages to the chat model.

        Args:
            new_messages (List[Dict[str, Any]]): list of messages. Each message is a dict with 'role', 'content', and 'function_call'.
        """
        # Save messages to Cloud DB
        config = read_config()
        if "dev_branch" in config and config["dev_branch"]["initializing"] == True:
            pass
        elif "dev_branch" in config and config["dev_branch"]["online"] == True:
            # if dev online=True, add to Local DB
            create_chat_logs(self.chat_uuid, new_messages, None)
        else:
            run_async_in_sync(
                self.llm_proxy._async_chat_log_to_cloud(
                    self.chat_uuid,
                    new_messages,
                    None,
                    {},
                )
            )

    def get_messages(self) -> List[Dict[str, Any]]:
        message_logs = run_async_in_sync(fetch_chat_log(self.chat_uuid))
        return message_logs

    def run(
        self,
        function_list: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = False,
    ) -> LLMResponse:
        """Run PromptModel. It does not raise error.

        Args:
            function_list (List[Dict[str, Any]], optional): list of functions to run. Defaults to None.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        # if stream:
        #     for item in self.llm_proxy.chat_stream(self.chat_uuid, function_list):
        #         yield item
        # else:
        #     return self.llm_proxy.chat_run(self.chat_uuid, function_list)
        return self.llm_proxy.chat_run(self.chat_uuid, function_list)

    async def arun(
        self,
        function_list: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = False,
    ) -> LLMResponse:
        """Async run PromptModel. It does not raise error.

        Args:
            function_list (List[Dict[str, Any]], optional): list of functions to run. Defaults to None.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        # if stream:
        #     return Coroutine(self.llm_proxy.chat_astream(self.chat_uuid, function_list))
        # else:
        #     return await self.llm_proxy.chat_arun(self.chat_uuid, function_list)
        return await self.llm_proxy.chat_arun(self.chat_uuid, function_list)
