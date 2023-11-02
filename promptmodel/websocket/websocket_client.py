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
from promptmodel import Client
from promptmodel.llms.llm_dev import LLMDev
from promptmodel.database.crud import (
    create_llm_module_version,
    create_prompt,
    create_run_log,
    list_llm_modules,
    list_llm_module_versions,
    list_prompts,
    list_run_logs,
    list_samples,
    get_sample_input,
    get_llm_module_uuid,
    update_llm_module_version,
    find_ancestor_version,
    find_ancestor_versions,
    update_candidate_version,
)
from promptmodel.database.models import LLMModuleVersion, LLMModule
from promptmodel.utils.enums import (
    ServerTask,
    LocalTask,
    LLMModuleVersionStatus,
)
from promptmodel.utils.types import LLMStreamResponse
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
    def __init__(self, _client: Client):
        self._client: Client = _client
        self.rwlock = rwlock.RWLockFair()

    async def _get_llm_modules(self, llm_module_name: str):
        """Get llm_module from registry"""
        with self.rwlock.gen_rlock():
            llm_module = next(
                (
                    llm_module
                    for llm_module in self._client.llm_modules
                    if llm_module.name == llm_module_name
                ),
                None,
            )
        return llm_module

    def update_client_instance(self, new_client):
        with self.rwlock.gen_wlock():
            self._client = new_client
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
            if message["type"] == LocalTask.LIST_MODULES:
                res_from_local_db = list_llm_modules()
                modules_with_local_usage = [
                    module
                    for module in res_from_local_db
                    if module["local_usage"] == True
                ]
                data = {"llm_modules": modules_with_local_usage}

            elif message["type"] == LocalTask.LIST_VERSIONS:
                llm_module_uuid = message["llm_module_uuid"]
                res_from_local_db = list_llm_module_versions(llm_module_uuid)
                data = {"llm_module_versions": res_from_local_db}

            elif message["type"] == LocalTask.LIST_SAMPLES:
                res_from_local_db = list_samples()
                data = {"samples": res_from_local_db}

            elif message["type"] == LocalTask.LIST_FUNCTIONS:
                function_name_list = self._client._get_function_name_list()
                data = {"functions": function_name_list}

            elif message["type"] == LocalTask.GET_PROMPTS:
                llm_module_version_uuid = message["llm_module_version_uuid"]
                res_from_local_db = list_prompts(llm_module_version_uuid)
                data = {"prompts": res_from_local_db}

            elif message["type"] == LocalTask.GET_RUN_LOGS:
                llm_module_version_uuid = message["llm_module_version_uuid"]
                res_from_local_db = list_run_logs(llm_module_version_uuid)
                data = {"run_logs": res_from_local_db}

            elif message["type"] == LocalTask.CHANGE_VERSION_STATUS:
                llm_module_version_uuid = message["llm_module_version_uuid"]
                new_status = message["status"]
                res_from_local_db = update_llm_module_version(
                    llm_module_version_uuid=llm_module_version_uuid, status=new_status
                )
                data = {
                    "llm_module_version_uuid": llm_module_version_uuid,
                    "status": new_status,
                }

            elif message["type"] == LocalTask.GET_VERSION_TO_SAVE:
                llm_module_version_uuid = message["llm_module_version_uuid"]
                # change from_uuid to candidate ancestor
                llm_module_version, prompts = find_ancestor_version(
                    llm_module_version_uuid
                )
                # delete status, candidate_version, is_published
                del llm_module_version["status"]
                del llm_module_version["candidate_version"]
                del llm_module_version["is_published"]

                for prompt in prompts:
                    del prompt["id"]

                llm_module = LLMModule.get(
                    LLMModule.uuid == llm_module_version.llm_module_uuid
                ).__data__
                is_deployed = llm_module["is_deployment"]
                if is_deployed:
                    data = {
                        "llm_module": {
                            "uuid": llm_module["uuid"],
                            "name": llm_module["name"],
                            "project_uuid": llm_module["project_uuid"],
                        },
                        "version": llm_module_version,
                        "prompts": prompts,
                    }
                else:
                    data = {
                        "llm_module": None,
                        "version": llm_module_version,
                        "prompts": prompts,
                    }

            elif message["type"] == LocalTask.GET_VERSIONS_TO_SAVE:
                target_llm_module_uuid = (
                    message["llm_module_uuid"] if "llm_module_uuid" in message else None
                )

                llm_module_versions, prompts = find_ancestor_versions(
                    target_llm_module_uuid
                )
                for llm_module_version in llm_module_versions:
                    del llm_module_version["id"]
                    del llm_module_version["status"]
                    del llm_module_version["candidate_version"]
                    del llm_module_version["is_published"]

                for prompt in prompts:
                    del prompt["id"]

                llm_module_uuids = [
                    version["llm_module_uuid"] for version in llm_module_versions
                ]
                llm_modules = list(
                    LLMModule.select().where(LLMModule.uuid.in_(llm_module_uuids))
                )
                llm_modules = [model_to_dict(llm_module) for llm_module in llm_modules]
                # find llm_module which is not deployed
                llm_modules_only_in_local = []
                for llm_module in llm_modules:
                    if llm_module["is_deployment"] is False:
                        del llm_module["is_deployment"]
                        del llm_module["local_usage"]
                        del llm_module["id"]
                        llm_modules_only_in_local.append(llm_module)

                data = {
                    "llm_modules": llm_modules_only_in_local,
                    "versions": llm_module_versions,
                    "prompts": prompts,
                }

            elif message["type"] == LocalTask.UPDATE_CANDIDATE_VERSION_ID:
                new_candidates = message["new_candidates"]
                update_candidate_version(new_candidates)

            elif message["type"] == LocalTask.RUN_LLM_MODULE:
                llm_module_name: str = message["llm_module_name"]
                sample_name: Optional[str] = message["sample_name"]

                # get sample from db
                if sample_name:
                    sample_input_row = get_sample_input(sample_name)
                    if sample_input_row is None or "contents" not in sample_input_row:
                        logger.error(f"There is no sample input {sample_name}.")
                        return
                    sample_input = sample_input_row["contents"]
                else:
                    sample_input = None

                # Check llm_module in Local Usage
                llm_module_names = self._client._get_llm_module_name_list()
                if llm_module_name not in llm_module_names:
                    logger.error(f"There is no llm_module {llm_module_name}.")
                    return

                # Validate Variable Matching
                prompt_variables = []
                for prompt in message["prompts"]:
                    fstring_input_pattern = r"(?<!\\)(?<!{{)\{([^}]+)\}(?!\\)(?!}})"
                    prompt_variables += re.findall(
                        fstring_input_pattern, prompt["content"]
                    )
                prompt_variables = list(set(prompt_variables))
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
                    logger.info(f"Started PromptModel: {llm_module_name}")
                    # create llm_module_dev_instance
                    llm_module_dev = LLMDev()
                    # fine llm_module_uuid from local db
                    llm_module_uuid: str = get_llm_module_uuid(llm_module_name)["uuid"]

                    llm_module_version_uuid: Optional[str] = message["uuid"]
                    # If llm_module_version_uuid is None, create new version & prompt
                    if llm_module_version_uuid is None:
                        llm_module_version: LLMModuleVersion = (
                            create_llm_module_version(
                                llm_module_uuid=llm_module_uuid,
                                status=LLMModuleVersionStatus.BROKEN.value,
                                from_uuid=message["from_uuid"],
                                model=message["model"],
                                parsing_type=message["parsing_type"],
                                output_keys=message["output_keys"],
                                functions=message["functions"],
                            )
                        )
                        llm_module_version_uuid: str = llm_module_version.uuid

                        prompts = message["prompts"]
                        for prompt in prompts:
                            create_prompt(
                                version_uuid=llm_module_version_uuid,
                                role=prompt["role"],
                                step=prompt["step"],
                                content=prompt["content"],
                            )
                        # send message to backend
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "llm_module_version_uuid": llm_module_version_uuid,
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

                    # get function descriptions from register & send to LLM
                    function_names: List[str] = message["functions"]
                    logger.debug(f"function_names : {function_names}")
                    try:
                        function_descriptions: List[
                            str
                        ] = self._client._get_function_descriptions(function_names)
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

                    logger.debug(f"functions : {function_descriptions}")

                    res: AsyncGenerator[
                        LLMStreamResponse, None
                    ] = llm_module_dev.dev_run(
                        messages=messages_for_run,
                        parsing_type=parsing_type,
                        functions=function_descriptions,
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
                            "name" : function_call['name'],
                            "arguments" : function_call['arguments'],
                            "response" : None,
                            "initial_raw_output" : output["raw_output"],
                        }

                        # call function
                        try:
                            function_call_args: Dict[str, Any] = json.loads(
                                function_call["arguments"]
                            )
                            function_response = self._client._call_register_function(
                                function_call["name"], function_call_args
                            )
                            function_call_log["response"] = function_response
                        except Exception as error:
                            logger.error(f"{error}")

                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "failed",
                                "log": f"Function call Failed, {error}",
                            }
                            update_llm_module_version(
                                llm_module_version_uuid=llm_module_version_uuid,
                                status=LLMModuleVersionStatus.BROKEN.value,
                            )
                            create_run_log(
                                llm_module_version_uuid=llm_module_version_uuid,
                                inputs=sample_input,
                                raw_output=output["raw_output"],
                                parsed_outputs=output["parsed_outputs"],
                                function_call=function_call_log,
                            )
                            response.update(data)
                            await ws.send(json.dumps(response, cls=CustomJSONEncoder))
                            return
                        # call LLM once more
                        messages_for_run.append(
                            {
                                "role": "function",
                                "name": function_call["name"],
                                "content": str(function_response),
                            }
                        )
                        res_after_function_call: AsyncGenerator[
                            LLMStreamResponse, None
                        ] = llm_module_dev.dev_chat(
                            messages_for_run, parsing_type, model
                        )

                        output = {"raw_output": "", "parsed_outputs": {}}
                        async for item in res_after_function_call:
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

                            if item.error and parsing_success is True:
                                parsing_success = not item.error
                                error_log = item.error_log

                            data.update(response)
                            # logger.debug(f"Sent response: {data}")
                            await ws.send(json.dumps(data, cls=CustomJSONEncoder))
                                
                    if (
                        message["output_keys"] is not None
                        and message['parsing_type'] is not None
                        and set(output["parsed_outputs"].keys()) != set(
                            message["output_keys"]
                        ) or (
                            parsing_success is False
                        )
                    ):
                        error_log = error_log if error_log else "Key matching failed."
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "status": "failed",
                            "log" : f"parsing failed, {error_log}"
                        }
                        update_llm_module_version(
                            llm_module_version_uuid=llm_module_version_uuid,
                            status=LLMModuleVersionStatus.BROKEN.value,
                        )
                        create_run_log(
                            llm_module_version_uuid=llm_module_version_uuid,
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
                    update_llm_module_version(
                        llm_module_version_uuid=llm_module_version_uuid,
                        status=LLMModuleVersionStatus.WORKING.value,
                    )

                    create_run_log(
                        llm_module_version_uuid=llm_module_version_uuid,
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
                    logger.success(
                        "Connected to gateway. Your Client Dev is now online! 🎉"
                    )
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
