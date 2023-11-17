import asyncio
import json
import datetime
import re

from uuid import UUID
from typing import Dict, Any, Optional, AsyncGenerator, List
from dotenv import load_dotenv

from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from readerwriterlock import rwlock
from playhouse.shortcuts import model_to_dict

import promptmodel.utils.logger as logger
from promptmodel import DevApp
from promptmodel.llms.llm_dev import LLMDev
from promptmodel.database.crud import (
    find_ancestor_version,
    find_ancestor_versions,
    update_candidate_prompt_model_version,
    update_candidate_chat_model_version,
    find_ancestor_chat_model_version,
    find_ancestor_chat_model_versions,
)
from promptmodel.database.models import *
from promptmodel.types.enums import (
    ServerTask,
    LocalTask,
    ModelVersionStatus,
)
from promptmodel.utils.config_utils import upsert_config, read_config
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
            if message["type"] == LocalTask.LIST_PROMPT_MODELS:
                prompt_model_rows = [
                    PromptModel.select().where(PromptModel.used_in_code == True)
                ]
                prompt_model_list = [
                    model_to_dict(model, recurse=False) for model in prompt_model_rows
                ]
                data = {"prompt_models": prompt_model_list}

            elif message["type"] == LocalTask.LIST_PROMPT_MODEL_VERSIONS:
                prompt_model_uuid = message["prompt_model_uuid"]
                prompt_model_version_rows = [
                    PromptModelVersion.select()
                    .where(PromptModelVersion.prompt_model_uuid == prompt_model_uuid)
                    .order_by(PromptModelVersion.created_at)
                ]
                prompt_model_version_list = [
                    model_to_dict(model, recurse=False)
                    for model in prompt_model_version_rows
                ]
                data = {"prompt_model_versions": prompt_model_version_list}

            elif message["type"] == LocalTask.LIST_SAMPLES:
                sample_inputs_rows = [SampleInputs.select()]
                sample_inputs_list = [
                    model_to_dict(model, recurse=False) for model in sample_inputs_rows
                ]
                data = {"samples": sample_inputs_list}

            elif message["type"] == LocalTask.LIST_FUNCTIONS:
                function_name_list = self._devapp._get_function_name_list()
                data = {"functions": function_name_list}

            elif message["type"] == LocalTask.GET_PROMPTS:
                prompt_model_version_uuid = message["prompt_model_version_uuid"]
                prompt_rows = [
                    Prompt.select()
                    .where(Prompt.version_uuid == prompt_model_version_uuid)
                    .order_by(Prompt.step)
                ]
                prompt_list = [
                    model_to_dict(model, recurse=False) for model in prompt_rows
                ]
                data = {"prompts": prompt_list}

            elif message["type"] == LocalTask.GET_RUN_LOGS:
                prompt_model_version_uuid = message["prompt_model_version_uuid"]
                run_log_rows = [
                    RunLog.select()
                    .where(RunLog.version_uuid == prompt_model_version_uuid)
                    .order_by(RunLog.created_at.desc())
                ]
                run_log_list = [
                    model_to_dict(model, recurse=False) for model in run_log_rows
                ]
                data = {"run_logs": run_log_list}

            elif message["type"] == LocalTask.CHANGE_PROMPT_MODEL_VERSION_STATUS:
                prompt_model_version_uuid = message["prompt_model_version_uuid"]
                new_status = message["status"]
                (
                    PromptModelVersion.update(status=new_status)
                    .where(PromptModelVersion.uuid == prompt_model_version_uuid)
                    .execute()
                )
                data = {
                    "prompt_model_version_uuid": prompt_model_version_uuid,
                    "status": new_status,
                }

            elif message["type"] == LocalTask.GET_PROMPT_MODEL_VERSION_TO_SAVE:
                prompt_model_version_uuid = message["prompt_model_version_uuid"]
                # change from_uuid to candidate ancestor
                prompt_model_version, prompts = find_ancestor_version(
                    prompt_model_version_uuid
                )
                # delete status, is_published
                del prompt_model_version["status"]
                del prompt_model_version["is_published"]
                config = read_config()
                prompt_model_version["dev_branch_uuid"] = config["dev_branch"]["uuid"]

                for prompt in prompts:
                    del prompt["id"]

                prompt_model = PromptModel.get(
                    PromptModel.uuid == prompt_model_version.prompt_model_uuid
                ).__data__
                is_deployed = prompt_model["is_deployed"]
                if not is_deployed:
                    data = {
                        "prompt_model": {
                            "uuid": prompt_model["uuid"],
                            "name": prompt_model["name"],
                            "project_uuid": prompt_model["project_uuid"],
                        },
                        "version": prompt_model_version,
                        "prompts": prompts,
                    }
                else:
                    data = {
                        "prompt_model": None,
                        "version": prompt_model_version,
                        "prompts": prompts,
                    }

            elif message["type"] == LocalTask.GET_PROMPT_MODEL_VERSIONS_TO_SAVE:
                target_prompt_model_uuid = (
                    message["prompt_model_uuid"]
                    if "prompt_model_uuid" in message
                    else None
                )

                prompt_model_versions, prompts = find_ancestor_versions(
                    target_prompt_model_uuid
                )

                config = read_config()
                for prompt_model_version in prompt_model_versions:
                    prompt_model_version["dev_branch_uuid"] = config["dev_branch"][
                        "uuid"
                    ]
                    del prompt_model_version["id"]
                    del prompt_model_version["status"]
                    del prompt_model_version["is_published"]

                for prompt in prompts:
                    del prompt["id"]

                prompt_model_uuids = [
                    version["prompt_model_uuid"] for version in prompt_model_versions
                ]
                prompt_models = list(
                    PromptModel.select().where(PromptModel.uuid.in_(prompt_model_uuids))
                )
                prompt_models = [
                    model_to_dict(prompt_model) for prompt_model in prompt_models
                ]
                # find prompt_model which is not deployed
                prompt_models_only_in_local = []
                for prompt_model in prompt_models:
                    if prompt_model["is_deployed"] is False:
                        del prompt_model["is_deployed"]
                        del prompt_model["used_in_code"]
                        del prompt_model["id"]

                        prompt_models_only_in_local.append(prompt_model)
                data = {
                    "prompt_models": prompt_models_only_in_local,
                    "versions": prompt_model_versions,
                    "prompts": prompts,
                }

            elif message["type"] == LocalTask.UPDATE_CANDIDATE_PROMPT_MODEL_VERSION_ID:
                new_candidates = message["new_candidates"]
                update_candidate_prompt_model_version(new_candidates)

            elif message["type"] == LocalTask.RUN_PROMPT_MODEL:
                prompt_model_name: str = message["prompt_model_name"]
                sample_name: Optional[str] = message["sample_name"]

                # get sample from db
                if sample_name:
                    try:
                        sample_input_row = SampleInputs.get(
                            SampleInputs.name == sample_name
                        )
                        sample_input_dict = model_to_dict(sample_input_row)
                    except:
                        sample_input_dict = None

                    if sample_input_dict is None or "contents" not in sample_input_dict:
                        logger.error(f"There is no sample input {sample_name}.")
                        return
                    sample_input = sample_input_dict["contents"]
                else:
                    sample_input = None

                # Check prompt_model in Local Usage
                prompt_model_names = self._devapp._get_prompt_model_name_list()
                if prompt_model_name not in prompt_model_names:
                    logger.error(f"There is no prompt_model {prompt_model_name}.")
                    return

                # Validate Variable Matching
                prompt_variables = []
                for prompt in message["prompts"]:
                    prompt_content = prompt["content"]
                    # Replace
                    escaped_patterns = re.findall(r"\{\{.*?\}\}", prompt_content)
                    for i, pattern in enumerate(escaped_patterns):
                        prompt_content = prompt_content.replace(
                            pattern, f"__ESCAPED{i}__"
                        )

                    # find f-string input variables
                    fstring_input_pattern = r"(?<!\\)\{([^}]+)\}(?<!\\})"
                    prompt_variables_in_prompt = re.findall(
                        fstring_input_pattern, prompt_content
                    )

                    # Replace back
                    for i, pattern in enumerate(escaped_patterns):
                        prompt_content = prompt_content.replace(
                            f"__ESCAPED{i}__", pattern
                        )

                    prompt_variables_in_prompt = list(set(prompt_variables_in_prompt))
                    prompt_variables += prompt_variables_in_prompt

                if len(prompt_variables) != 0:
                    if sample_input is None:
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "status": "failed",
                            "log": f"Prompts have variables {prompt_variables}. You should select sample input.",
                        }
                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                        return

                    if not all(
                        variable in sample_input.keys() for variable in prompt_variables
                    ):
                        missing_variables = [
                            variable
                            for variable in prompt_variables
                            if variable not in sample_input.keys()
                        ]
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "status": "failed",
                            "log": f"Sample input does not have variables {missing_variables} in prompts.",
                        }
                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                        return

                # Start PromptModel Running
                output = {"raw_output": "", "parsed_outputs": {}}
                try:
                    logger.info(f"Started PromptModel: {prompt_model_name}")
                    # create prompt_model_dev_instance
                    prompt_model_dev = LLMDev()
                    # fine prompt_model_uuid from local db
                    prompt_model_row = PromptModel.get(
                        PromptModel.name == prompt_model_name
                    )
                    prompt_model_uuid: str = model_to_dict(
                        prompt_model_row, recurse=False
                    )["uuid"]

                    prompt_model_version_uuid: Optional[str] = message["uuid"]
                    # If prompt_model_version_uuid is None, create new version & prompt
                    if prompt_model_version_uuid is None:
                        prompt_model_version: PromptModelVersion = (
                            PromptModelVersion.create(
                                prompt_model_uuid=prompt_model_uuid,
                                status=ModelVersionStatus.BROKEN.value,
                                from_uuid=message["from_uuid"],
                                model=message["model"],
                                parsing_type=message["parsing_type"],
                                output_keys=message["output_keys"],
                                functions=message["functions"],
                            )
                        )
                        prompt_model_version_uuid: str = prompt_model_version.uuid

                        prompts = message["prompts"]
                        prompt_rows_to_insert = [
                            {
                                "version_uuid": prompt_model_version_uuid,
                                "role": prompt["role"],
                                "step": prompt["step"],
                                "content": prompt["content"],
                            }
                            for prompt in prompts
                        ]
                        Prompt.insert_many(prompt_rows_to_insert).execute()

                        # send message to backend
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "prompt_model_version_uuid": prompt_model_version_uuid,
                            "status": "running",
                        }
                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "running",
                        "inputs": sample_input if sample_input else {},
                    }
                    data.update(response)
                    # logger.debug(f"Sent response: {data}")
                    await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    model = message["model"]
                    prompts = message["prompts"]
                    parsing_type = message["parsing_type"]

                    if sample_input:
                        messages_for_run = [
                            {
                                "content": prompt["content"].format(**sample_input),
                                "role": prompt["role"],
                            }
                            for prompt in prompts
                        ]
                    else:
                        messages_for_run = prompts

                    parsing_success = True
                    error_log = None
                    function_call = None
                    function_call_log = None

                    # get function schemas from register & send to LLM
                    function_names: List[str] = message["functions"]
                    logger.debug(f"function_names : {function_names}")
                    try:
                        function_schemas: List[
                            str
                        ] = self._devapp._get_function_schemas(function_names)
                    except Exception as error:
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "status": "failed",
                            "log": f"There is no function. : {error}",
                        }
                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                        return

                    logger.debug(f"functions : {function_schemas}")

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
                            if (
                                list(item.parsed_outputs.keys())[0]
                                not in output["parsed_outputs"]
                            ):
                                output["parsed_outputs"][
                                    list(item.parsed_outputs.keys())[0]
                                ] = list(item.parsed_outputs.values())[0]
                            else:
                                output["parsed_outputs"][
                                    list(item.parsed_outputs.keys())[0]
                                ] += list(item.parsed_outputs.values())[0]
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "parsed_outputs": item.parsed_outputs,
                            }
                        if item.function_call is not None:
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "function_call": item.function_call,
                            }
                            function_call = item.function_call

                        if item.error and parsing_success is True:
                            parsing_success = not item.error
                            error_log = item.error_log

                        data.update(response)
                        # logger.debug(f"Sent response: {data}")
                        await ws.send(json.dumps(data, cls=CustomJSONEncoder))

                    # IF function_call in response -> call function -> call LLM once more
                    if function_call:
                        # make function_call_log
                        function_call_log = {
                            "name": function_call["name"],
                            "arguments": function_call["arguments"],
                            "response": None,
                            "initial_raw_output": output["raw_output"],
                        }

                        # call function
                        try:
                            function_call_args: Dict[str, Any] = json.loads(
                                function_call["arguments"]
                            )
                            function_response = self._devapp._call_register_function(
                                function_call["name"], function_call_args
                            )
                            function_call_log["response"] = function_response
                            # Send function call response for check LLM response validity
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "function_response": function_response,
                            }
                            data.update(response)
                            # logger.debug(f"Sent response: {data}")
                            await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                        except Exception as error:
                            logger.error(f"{error}")

                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "failed",
                                "log": f"Function call Failed, {error}",
                            }

                            (
                                PromptModelVersion.update(
                                    status=ModelVersionStatus.BROKEN.value
                                )
                                .where(
                                    PromptModelVersion.uuid == prompt_model_version_uuid
                                )
                                .execute()
                            )

                            RunLog.create(
                                version_uuid=prompt_model_version_uuid,
                                inputs=sample_input,
                                raw_output=output["raw_output"],
                                parsed_outputs=output["parsed_outputs"],
                                function_call=function_call_log,
                            )
                            response.update(data)
                            await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                            return
                        # # call LLM once more
                        # messages_for_run += [
                        #     {
                        #         "role": "assistant",
                        #         "function_call": function_call,
                        #     },
                        #     {
                        #         "role": "function",
                        #         "name": function_call["name"],
                        #         "content": str(function_response),
                        #     },
                        # ]

                        # res_after_function_call: AsyncGenerator[
                        #     LLMStreamResponse, None
                        # ] = prompt_model_dev.dev_chat(
                        #     messages_for_run, parsing_type, model
                        # )

                        # output = {"raw_output": "", "parsed_outputs": {}}
                        # async for item in res_after_function_call:
                        #     if item.raw_output is not None:
                        #         output["raw_output"] += item.raw_output
                        #         data = {
                        #             "type": ServerTask.UPDATE_RESULT_RUN.value,
                        #             "status": "running",
                        #             "raw_output": item.raw_output,
                        #         }
                        #     if item.parsed_outputs:
                        #         if (
                        #             list(item.parsed_outputs.keys())[0]
                        #             not in output["parsed_outputs"]
                        #         ):
                        #             output["parsed_outputs"][
                        #                 list(item.parsed_outputs.keys())[0]
                        #             ] = list(item.parsed_outputs.values())[0]
                        #         else:
                        #             output["parsed_outputs"][
                        #                 list(item.parsed_outputs.keys())[0]
                        #             ] += list(item.parsed_outputs.values())[0]
                        #         data = {
                        #             "type": ServerTask.UPDATE_RESULT_RUN.value,
                        #             "status": "running",
                        #             "parsed_outputs": item.parsed_outputs,
                        #         }

                        #     if item.error and parsing_success is True:
                        #         parsing_success = not item.error
                        #         error_log = item.error_log

                        #     data.update(response)
                        #     # logger.debug(f"Sent response: {data}")
                        #     await ws.send(json.dumps(data, cls=CustomJSONEncoder))

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
                            "log": f"parsing failed, {error_log}",
                        }

                        (
                            PromptModelVersion.update(
                                status=ModelVersionStatus.BROKEN.value
                            )
                            .where(PromptModelVersion.uuid == prompt_model_version_uuid)
                            .execute()
                        )

                        RunLog.create(
                            version_uuid=prompt_model_version_uuid,
                            inputs=sample_input,
                            raw_output=output["raw_output"],
                            parsed_outputs=output["parsed_outputs"],
                            function_call=function_call_log,
                        )
                        response.update(data)
                        await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                        return

                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "completed",
                    }

                    (
                        PromptModelVersion.update(
                            status=ModelVersionStatus.WORKING.value
                        )
                        .where(PromptModelVersion.uuid == prompt_model_version_uuid)
                        .execute()
                    )

                    RunLog.create(
                        version_uuid=prompt_model_version_uuid,
                        inputs=sample_input,
                        raw_output=output["raw_output"],
                        parsed_outputs=output["parsed_outputs"],
                        function_call=function_call_log,
                    )
                except Exception as error:
                    logger.error(f"Error running service: {error}")
                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "failed",
                        "log": str(error),
                    }
                    response.update(data)
                    await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                    return

            # ChatModel LocalTasks
            elif message["type"] == LocalTask.CHANGE_CHAT_MODEL_VERSION_STATUS:
                chat_model_version_uuid = message["chat_model_version_uuid"]
                new_status = message["status"]
                (
                    ChatModelVersion.update(status=new_status)
                    .where(ChatModelVersion.uuid == chat_model_version_uuid)
                    .execute()
                )

                data = {
                    "chat_model_version_uuid": chat_model_version_uuid,
                    "status": new_status,
                }

            elif message["type"] == LocalTask.GET_CHAT_LOG_SESSIONS:
                chat_model_version_uuid = message["chat_model_version_uuid"]
                session_rows = [
                    ChatLogSession.select()
                    .where(ChatLogSession.version_uuid == chat_model_version_uuid)
                    .order_by(ChatLogSession.created_at.desc())
                ]
                session_list = [
                    model_to_dict(model, recurse=False) for model in session_rows
                ]

                data = {"sessions": session_list}

            elif message["type"] == LocalTask.GET_CHAT_LOGS:
                session_uuid = message["session_uuid"]
                chat_log_rows = [
                    ChatLog.select()
                    .where(ChatLog.session_uuid == session_uuid)
                    .order_by(ChatLog.created_at.asc())
                ]
                chat_log_list = [
                    model_to_dict(model, recurse=False) for model in chat_log_rows
                ]
                data = {"chat_logs": chat_log_list}

            elif message["type"] == LocalTask.GET_CHAT_MODEL_VERSION_TO_SAVE:
                chat_model_version_uuid = message["chat_model_version_uuid"]
                # change from_uuid to candidate ancestor
                chat_model_version = find_ancestor_chat_model_version(
                    chat_model_version_uuid
                )
                # delete status, is_published
                del chat_model_version["status"]
                del chat_model_version["is_published"]
                config = read_config()
                chat_model_version["dev_branch_uuid"] = config["dev_branch"]["uuid"]

                chat_model = ChatModel.get(
                    ChatModel.uuid == chat_model_version.chat_model_uuid
                ).__data__

                is_deployed = chat_model["is_deployed"]
                if not is_deployed:
                    data = {
                        "chat_model": {
                            "uuid": chat_model["uuid"],
                            "name": chat_model["name"],
                            "project_uuid": chat_model["project_uuid"],
                        },
                        "version": chat_model_version,
                    }
                else:
                    data = {
                        "chat_model": None,
                        "version": chat_model_version,
                    }

            elif message["type"] == LocalTask.GET_CHAT_MODEL_VERSIONS_TO_SAVE:
                target_chat_model_uuid = (
                    message["chat_model_uuid"] if "chat_model_uuid" in message else None
                )

                chat_model_versions = find_ancestor_chat_model_versions(
                    target_chat_model_uuid
                )

                config = read_config()
                for chat_model_version in chat_model_versions:
                    chat_model_version["dev_branch_uuid"] = config["dev_branch"]["uuid"]
                    del chat_model_version["id"]
                    del chat_model_version["status"]
                    del chat_model_version["is_published"]

                chat_model_uuids = [
                    version["chat_model_uuid"] for version in chat_model_versions
                ]
                chat_models = list(
                    ChatModel.select().where(ChatModel.uuid.in_(chat_model_uuids))
                )
                chat_models = [model_to_dict(chat_model) for chat_model in chat_models]
                # find chat_model which is not deployed
                chat_models_only_in_local = []
                for chat_model in chat_models:
                    if chat_model["is_deployed"] is False:
                        del chat_model["is_deployed"]
                        del chat_model["used_in_code"]
                        del chat_model["id"]

                        chat_models_only_in_local.append(chat_model)
                data = {
                    "chat_models": chat_models_only_in_local,
                    "versions": chat_model_versions,
                }

            elif message["type"] == LocalTask.UPDATE_CANDIDATE_CHAT_MODEL_VERSION_ID:
                new_candidates = message["new_candidates"]
                update_candidate_chat_model_version(new_candidates)

            elif message["type"] == LocalTask.RUN_CHAT_MODEL:
                pass

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
        dev_branch_name: str,
        cli_access_header: dict,
        retries=12 * 24,
        retry_delay=5 * 60,
    ):
        """Open Websocket to Backend with project_uuid, dev_branch_name, cli_access_token"""
        headers = cli_access_header
        headers.update(
            {"project_uuid": project_uuid, "dev_branch_name": dev_branch_name}
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
