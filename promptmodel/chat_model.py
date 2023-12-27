from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Coroutine, Union
from uuid import uuid4

from litellm import ModelResponse

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
from promptmodel.types.enums import InstanceType
from promptmodel.types.request import ChatLogRequest
from promptmodel.apis.base import AsyncAPIClient


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
        version (Optional[ Union[str, int] ], optional): Choose which FunctionModel version to use. Defaults to "deploy". It can be "deploy", "latest", or version number.
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
        self.recent_log_uuid = None

        if session_uuid is None:
            self.session_uuid = str(uuid4())
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

    @check_connection_status_decorator
    def get_config(
        self,
        *args,
        **kwargs,
    ) -> ChatModelConfig:
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

        return ChatModelConfig(
            system_prompt=prompt,
            model=version_detail["model"],
            name=self.name,
            version_uuid=str(version_detail["uuid"]),
            version=version_detail["version"],
            message_logs=message_logs,
        )

    @check_connection_status_decorator
    def add_messages(
        self,
        new_messages: List[Dict[str, Any]],
        metadata_list: List[Optional[Dict]] = [],
        *args,
        **kwargs,
    ) -> None:
        """Add messages to the chat model.

        Args:
            new_messages (List[Dict[str, Any]]): list of messages. Each message is a dict with 'role', 'content', and 'function_call'.
        """
        # Save messages to Cloud DB
        log_uuid_list = [str(uuid4()) for _ in range(len(new_messages))]
        run_async_in_sync(
            self.llm_proxy._async_chat_log_to_cloud(
                session_uuid=str(self.session_uuid),
                version_uuid=None,
                chat_log_request_list=[
                    ChatLogRequest(**{"message": message, "uuid": str(uuid4())})
                    for message in new_messages
                ],
            )
        )

        self.recent_log_uuid = log_uuid_list[-1]

    @check_connection_status_decorator
    def run(
        self,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Run FunctionModel. It does not raise error.

        Args:
            functions (List[Dict[str, Any]], optional): list of functions to run. Defaults to None.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        if stream:

            def gen():
                cache: Optional[LLMStreamResponse] = None

                for item in self.llm_proxy.chat_stream(
                    self.session_uuid, functions, tools
                ):
                    yield item
                    cache: LLMStreamResponse = item
                if cache:
                    self.recent_log_uuid = cache.pm_detail.log_uuid

            return gen()
        else:
            res = self.llm_proxy.chat_run(self.session_uuid, functions, tools)
            self.recent_log_uuid = res.pm_detail.log_uuid
            return res
        # return self.llm_proxy.chat_run(self.session_uuid, functions, self.api_key)

    @check_connection_status_decorator
    async def arun(
        self,
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Async run FunctionModel. It does not raise error.

        Args:
            functions (List[Dict[str, Any]], optional): list of functions to run. Defaults to None.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        if stream:

            async def async_gen():
                cache: Optional[LLMStreamResponse] = None

                async for item in self.llm_proxy.chat_astream(
                    self.session_uuid, functions, tools
                ):
                    yield item
                    cache: LLMStreamResponse = item
                if cache:
                    self.recent_log_uuid = cache.pm_detail.log_uuid

            return async_gen()
        else:
            res: LLMResponse = await self.llm_proxy.chat_arun(
                self.session_uuid, functions
            )
            self.recent_log_uuid = res.pm_detail.log_uuid
            return res
        # return await self.llm_proxy.chat_arun(
        #     self.session_uuid, functions, self.api_key
        # )

    @check_connection_status_decorator
    async def log_score_to_session(
        self, score: Optional[Dict[str, Any]] = {}, *args, **kwargs
    ):
        try:
            res = await AsyncAPIClient.execute(
                method="POST",
                path="/save_chat_session_score",
                params={
                    "chat_session_uuid": self.session_uuid,
                },
                json=score,
                use_cli_key=False,
            )
            if res.status_code != 200:
                logger.error(f"Logging error: {res}")
        except Exception as exception:
            logger.error(f"Logging error: {exception}")

    @check_connection_status_decorator
    async def log_score(
        self,
        log_uuid: Optional[str] = None,
        score: Optional[Dict[str, Any]] = {},
        *args,
        **kwargs,
    ):
        """Log score for the ChatMessage in the Cloud DB.

        Args:
            log_uuid (Optional[str], optional): UUID of log to save score. Defaults to None. If None, it will use the recent log_uuid.
            score (Optional[Dict[str, Any]], optional): scores for message. Each keys will be created as Evaluation Metric in Promptmodel Dashboard. Defaults to {}.
        """
        try:
            if not log_uuid and self.recent_log_uuid:
                log_uuid = self.recent_log_uuid

            if log_uuid is None:
                raise ValueError(
                    "log_uuid is None. Please give log_uuid or run ChatModel.log or ChatModel.run first."
                )

            res = await AsyncAPIClient.execute(
                method="POST",
                path="/chat_message_score",
                params={
                    "chat_message_uuid": log_uuid,
                },
                json=score,
                use_cli_key=False,
            )
            if res.status_code != 200:
                logger.error(f"Logging error: {res}")
        except Exception as exception:
            logger.error(f"Logging error: {exception}")

    @check_connection_status_decorator
    async def log(
        self,
        openai_api_response: Optional[ModelResponse] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> List[str]:
        """
        Log Messages to the Cloud DB.
        Args:
            openai_api_response (Optional[ModelResponse], optional): Response from the LLM. Defaults to None.
            messages (Optional[List[Dict[str, Any]]], optional): List of messages. Defaults to None.
            metadata (Optional[Dict[str, Any]], optional): Metadata for the log. Defaults to {}. It will be saved in the Last Log.

        You can give either openai_api_response or messages.
        If you give messages, it will log messages to the Cloud DB. This is useful when you want to log user messages to the Cloud DB.
        If you give openai_api_response, it will log messages to the Cloud DB. This is useful when you want to log the response of the LLM to the Cloud DB.

        Returns:
            List[str]: List of log_uuids in the Cloud DB. You can log scores for each logs using this log_uuid with method ChatModel.log_score.
        """
        try:
            if messages and openai_api_response:
                raise ValueError(
                    "You can give either openai_api_response or messages. You cannot give both."
                )

            chat_message_requests_body = []
            if messages:
                chat_message_requests_body = [
                    ChatLogRequest(
                        **{"message": message, "uuid": str(uuid4())}
                    ).model_dump()
                    for message in messages
                ]
            elif openai_api_response:
                chat_message_requests_body = [
                    ChatLogRequest(
                        **{
                            "message": openai_api_response.choices[
                                0
                            ].message.model_dump(),
                            "uuid": str(uuid4()),
                            "api_response": openai_api_response,
                        }
                    ).model_dump()
                ]

            chat_message_requests_body[-1]["metadata"] = metadata

            uuid_list = [x["uuid"] for x in chat_message_requests_body]

            res = await AsyncAPIClient.execute(
                method="POST",
                path="/chat_log",
                params={
                    "session_uuid": self.session_uuid,
                },
                json=chat_message_requests_body,
                use_cli_key=False,
            )
            if res.status_code != 200:
                logger.error(f"Logging error: {res}")

            # last log_uuid will be saved in the recent_log_uuid
            self.recent_log_uuid = uuid_list[-1]
            return uuid_list
        except Exception as exception:
            logger.error(f"Logging error: {exception}")
