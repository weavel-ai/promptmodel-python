import asyncio
import json
import datetime
import re

from uuid import UUID, uuid4
from typing import Dict, Any, Optional, AsyncGenerator, List
from dotenv import load_dotenv
from collections import defaultdict
from asyncio import Queue

from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from readerwriterlock import rwlock
from playhouse.shortcuts import model_to_dict

import promptmodel.utils.logger as logger
from promptmodel import DevApp
from promptmodel.llms.llm_dev import LLMDev
from promptmodel.database.models import (
    DeployedPromptModel,
    DeployedPromptModelVersion,
    DeployedPrompt,
)
from promptmodel.types.enums import ServerTask, LocalTask, LocalTaskErrorType
from promptmodel.utils.config_utils import upsert_config, read_config
from promptmodel.utils.output_utils import update_dict
from promptmodel.types.response import LLMStreamResponse
from promptmodel.constants import ENDPOINT_URL

load_dotenv()
GATEWAY_URL = f"wss://{ENDPOINT_URL.split('://')[1]}/open_websocket"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            aware_datetime = obj.replace(tzinfo=datetime.timezone.utc)
            return aware_datetime.isoformat()  # This will include timezone information
        return super().default(obj)


class DevWebsocketClient:
    def __init__(self, _devapp: DevApp):
        self._devapp: DevApp = _devapp
        self.rwlock = rwlock.RWLockFair()
        self.pending_requests: Dict[str, asyncio.Event] = {}
        self.responses: Dict[str, Queue] = defaultdict(Queue)

    async def _get_prompt_models(self, prompt_model_name: str):
        """Get prompt_model from registry"""
        with self.rwlock.gen_rlock():
            prompt_model = next(
                (
                    prompt_model
                    for prompt_model in self._devapp.prompt_models
                    if prompt_model.name == prompt_model_name
                ),
                None,
            )
        return prompt_model

    def update_devapp_instance(self, new_devapp):
        with self.rwlock.gen_wlock():
            self._devapp = new_devapp
            if self.ws:
                asyncio.run(
                    self.ws.send(
                        json.dumps({"type": ServerTask.LOCAL_UPDATE_ALERT.value})
                    )
                )

    async def __handle_message(
        self, message: Dict[str, Any], ws: WebSocketClientProtocol
    ):
        # logger.info(f"Received message: {message}")
        response: Dict[Any, str] = {}
        # If the message has a correlation_id, add it to the response
        # correlation_id is the unique ID of the function from backend to local
        if message.get("correlation_id"):
            response["correlation_id"] = message["correlation_id"]

        # If the message has a runner_id, add it to the response
        if message.get("runner_id"):
            response["runner_id"] = message["runner_id"]

        data = None
        try:
            if message["type"] == LocalTask.RUN_PROMPT_MODEL:
                messages: List[Dict] = message["messages_for_run"]

                # # Check prompt_model in Local Usage
                # prompt_model_names = self._devapp._get_prompt_model_name_list()
                # if prompt_model_name not in prompt_model_names:
                #     logger.error(f"There is no prompt_model {prompt_model_name}.")
                #     return

                # Start PromptModel Running
                output = {"raw_output": "", "parsed_outputs": {}}
                try:
                    logger.info("Started PromptModel")
                    # create prompt_model_dev_instance
                    prompt_model_dev = LLMDev()
                    # find prompt_model_uuid from local db

                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "running",
                    }
                    data.update(response)
                    # logger.debug(f"Sent response: {data}")
                    await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    model = message["model"]
                    parsing_type = message["parsing_type"]

                    messages_for_run = messages

                    parsing_success = True
                    error_log = None
                    function_call = None
                    function_schemas: List[Dict] = message[
                        "function_schemas"
                    ]  # this has a mock_response which should not be sent to LLM
                    function_mock_responses = {}
                    for function_schema in function_schemas:
                        function_mock_responses[
                            function_schema["name"]
                        ] = function_schema["mock_response"]

                    for schema in function_schemas:
                        del schema["mock_response"]

                    res: AsyncGenerator[
                        LLMStreamResponse, None
                    ] = prompt_model_dev.dev_run(
                        messages=messages_for_run,
                        parsing_type=parsing_type,
                        functions=function_schemas,
                        model=model,
                    )

                    async for item in res:
                        # send item to backend
                        # save item & parse
                        # if type(item) == str: raw output, if type(item) == dict: parsed output
                        if item.raw_output is not None:
                            output["raw_output"] += item.raw_output
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "raw_output": item.raw_output,
                            }
                        if item.parsed_outputs:
                            output["parsed_outputs"] = update_dict(
                                output["parsed_outputs"], item.parsed_outputs
                            )

                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "parsed_outputs": item.parsed_outputs,
                            }
                        if item.function_call is not None:
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "function_call": item.function_call.model_dump(),
                            }
                            function_call = item.function_call.model_dump()

                        if item.error and parsing_success is True:
                            parsing_success = not item.error
                            error_log = item.error_log

                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    # IF function_call in response -> call function
                    if function_call:
                        if (
                            function_call["name"]
                            in self._devapp._get_function_name_list()
                        ):
                            # call function
                            try:
                                function_call_args: Dict[str, Any] = json.loads(
                                    function_call["arguments"]
                                )
                                function_response = (
                                    self._devapp._call_register_function(
                                        function_call["name"], function_call_args
                                    )
                                )

                                # Send function call response for check LLM response validity
                                data = {
                                    "type": ServerTask.UPDATE_RESULT_RUN.value,
                                    "status": "running",
                                    "function_response": {
                                        "name": function_call["name"],
                                        "response": function_response,
                                    },
                                }
                                data.update(response)
                                # logger.debug(f"Sent response: {data}")
                                await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                            except Exception as error:
                                logger.error(f"{error}")

                                data = {
                                    "type": ServerTask.UPDATE_RESULT_RUN.value,
                                    "status": "failed",
                                    "error_type": LocalTaskErrorType.FUNCTION_CALL_FAILED_ERROR.value,
                                    "log": f"Function call Failed, {error}",
                                }

                                response.update(data)
                                await ws.send(
                                    json.dumps(response, cls=CustomJSONEncoder)
                                )
                                return

                        else:
                            # return mock response
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "function_response": {
                                    "name": function_call["name"],
                                    "response": "FAKE RESPONSE : "
                                    + str(
                                        function_mock_responses[function_call["name"]]
                                    ),
                                },
                            }
                            data.update(response)
                            # logger.debug(f"Sent response: {data}")
                            await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    if (
                        message["output_keys"] is not None
                        and message["parsing_type"] is not None
                        and set(output["parsed_outputs"].keys())
                        != set(
                            message["output_keys"]
                        )  # parsed output keys != output keys
                    ) or (
                        parsing_success is False
                    ):  # error occurs in streaming time
                        error_log = error_log if error_log else "Key matching failed."
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "status": "failed",
                            "error_type": LocalTaskErrorType.PARSING_FAILED_ERROR.value,
                            "log": f"parsing failed, {error_log}",
                        }
                        response.update(data)
                        await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                        return

                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "completed",
                    }

                except Exception as error:
                    logger.error(f"Error running service: {error}")
                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "failed",
                        "error_type": LocalTaskErrorType.SERVICE_ERROR.value,
                        "log": str(error),
                    }
                    response.update(data)
                    await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                    return

            elif message["type"] == LocalTask.RUN_CHAT_MODEL:
                old_messages = message["old_messages"]
                new_messages = message["new_messages"]
                messages_for_run = old_messages + new_messages
                # Start ChatModel Running
                try:
                    logger.info("Started ChatModel")
                    chat_model_dev = LLMDev()

                    messages_for_run = old_messages + new_messages

                    error_log = None
                    function_call = None

                    function_schemas: List[Dict] = message[
                        "function_schemas"
                    ]  # this has a mock_response which should not be sent to LLM
                    function_mock_responses = {}
                    for function_schema in function_schemas:
                        function_mock_responses[
                            function_schema["name"]
                        ] = function_schema["mock_response"]

                    for schema in function_schemas:
                        del schema["mock_response"]

                    res: AsyncGenerator[
                        LLMStreamResponse, None
                    ] = chat_model_dev.dev_chat(
                        messages=messages_for_run,
                        functions=function_schemas,
                        model=message["model"],
                    )

                    raw_output = ""
                    logger.debug(f"Mock responses: {function_mock_responses}")
                    logger.debug(f"Res: {res}")
                    async for chunk in res:
                        if chunk.raw_output is not None:
                            logger.debug(f"Chunk: {chunk}")
                            raw_output += chunk.raw_output
                            data = {
                                "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                                "status": "running",
                                "raw_output": chunk.raw_output,
                            }
                        logger.debug(f"Chunk: {chunk}")
                        if chunk.function_call is not None:
                            data = {
                                "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                                "status": "running",
                                "function_call": chunk.function_call.model_dump(),
                            }
                            if function_call is None:
                                function_call = {}
                            function_call = update_dict(
                                function_call, chunk.function_call.model_dump()
                            )

                        if chunk.error:
                            error_log = chunk.error_log
                        logger.debug(f"Response: {response}")
                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                    # IF function_call in response -> call function -> call LLM once more
                    logger.debug(f"Function call: {function_call}")

                    if function_call is not None:
                        if (
                            function_call["name"]
                            in self._devapp._get_function_name_list()
                        ):
                            # call function
                            try:
                                function_call_args: Dict[str, Any] = json.loads(
                                    function_call["arguments"]
                                )
                                function_response = (
                                    self._devapp._call_register_function(
                                        function_call["name"], function_call_args
                                    )
                                )

                                # Send function call response for check LLM response validity
                                data = {
                                    "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                                    "status": "running",
                                    "function_response": {
                                        "name": function_call["name"],
                                        "response": function_response,
                                    },
                                }
                                data.update(response)
                                await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                                logger.debug(f"Sent response: {data}")
                            except Exception as error:
                                logger.error(f"{error}")

                                data = {
                                    "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                                    "status": "failed",
                                    "error_type": LocalTaskErrorType.FUNCTION_CALL_FAILED_ERROR.value,
                                    "log": f"Function call Failed, {error}",
                                }

                                response.update(data)
                                await ws.send(
                                    json.dumps(response, cls=CustomJSONEncoder)
                                )
                                return
                        else:
                            # return mock response
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "function_response": {
                                    "name": function_call["name"],
                                    "response": "FAKE RESPONSE : "
                                    + function_mock_responses[function_call["name"]],
                                },
                            }
                            data.update(response)
                            # logger.debug(f"Sent response: {data}")
                            await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                            function_response = function_mock_responses[
                                function_call["name"]
                            ]

                        # call LLM once more
                        messages_for_run += [
                            {
                                "role": "assistant",
                                "content": "",
                                "function_call": function_call,
                            },
                            {
                                "role": "function",
                                "name": function_call["name"],
                                "content": str(function_response),
                            },
                        ]

                        res_after_function_call: AsyncGenerator[
                            LLMStreamResponse, None
                        ] = chat_model_dev.dev_chat(
                            messages=messages_for_run,
                            model=message["model"],
                        )

                        raw_output = ""
                        async for item in res_after_function_call:
                            if item.raw_output is not None:
                                raw_output += item.raw_output
                                data = {
                                    "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                                    "status": "running",
                                    "raw_output": item.raw_output,
                                }

                            if item.error:
                                error_log = item.error_log

                            data.update(response)
                            # logger.debug(f"Sent response: {data}")
                            await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    data = {
                        "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                        "status": "completed",
                    }

                except Exception as error:
                    logger.error(f"Error running service: {error}")
                    data = {
                        "type": ServerTask.UPDATE_RESULT_CHAT_RUN.value,
                        "status": "failed",
                        "error_type": LocalTaskErrorType.SERVICE_ERROR.value,
                        "log": str(error),
                    }
                    response.update(data)
                    await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                    return

            if data:
                response.update(data)
                await ws.send(json.dumps(response, cls=CustomJSONEncoder))
            logger.info(f"Sent response: {response}")
        except Exception as error:
            logger.error(f"Error handling message: {error}")
            await ws.send(str(error))

    async def connect_to_gateway(
        self,
        project_uuid: str,
        connection_name: str,
        cli_access_header: dict,
        retries=12 * 24,
        retry_delay=5 * 60,
    ):
        """Open Websocket to Backend with project_uuid, connection_name, cli_access_token"""
        headers = cli_access_header
        headers.update(
            {"project_uuid": project_uuid, "connection_name": connection_name}
        )
        for _ in range(retries):
            try:
                async with connect(
                    GATEWAY_URL,
                    extra_headers=headers,
                    # ping_interval=10,
                    # ping_timeout=1,
                    # timeout=3600 * 24,  # Timeout is set to 24 hours
                ) as ws:
                    logger.success("Connected to gateway. Your DevApp is now online! 🎉")
                    self.ws = ws
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        correlation_id = data.get("correlation_id")

                        if correlation_id and correlation_id in self.pending_requests:
                            await self.responses[correlation_id].put(data)
                            if not self.pending_requests[correlation_id].is_set():
                                self.pending_requests[
                                    correlation_id
                                ].set()  # Signal the event that the response has arrived
                        else:
                            await self.__handle_message(data, ws)
            except (ConnectionClosedError, ConnectionClosedOK):
                # If the connection was closed gracefully, handle it accordingly
                logger.warning("Connection to the gateway was closed.")
            except TimeoutError:
                logger.error(
                    f"Timeout error while connecting to the gateway. Retrying in {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)
            except Exception as error:
                logger.error(f"Error receiving message: {error}")

    async def request(self, type: ServerTask, message: Dict = {}):
        """
        Send a message to the connected server and wait for a response.

        Returns a python object.
        """
        ws = self.ws
        if ws:
            correlation_id = str(uuid4())  # Generate unique correlation ID
            message["correlation_id"] = correlation_id

            try:
                message["type"] = type.value
                await ws.send(json.dumps(message))
                logger.success(
                    f"""Sent request to local.
    - Message: {message}"""
                )
                event = asyncio.Event()
                self.pending_requests[correlation_id] = event

                await asyncio.wait_for(event.wait(), timeout=120)  # 2 minutes timeout
                response = await self.responses[correlation_id].get()
                logger.debug(response)
                return response
            except Exception as error:
                logger.error(
                    f"""Error for request to local: {error}
    - Message: {message}"""
                )
            finally:
                self.pending_requests.pop(correlation_id, None)
                self.responses.pop(correlation_id, None)
        else:
            raise ValueError(f"No active connection found")
