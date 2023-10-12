import time
import asyncio
from ..constants import ENDPOINT_URL
import typer
import json
import sys
import os
import importlib
import datetime

from uuid import UUID
from requests import request
from typing import Callable, Dict, Any, List
from dotenv import load_dotenv
from threading import Timer

import webbrowser
import inspect

from rich import print
from InquirerPy import inquirer
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from readerwriterlock import rwlock
from watchdog.events import FileSystemEventHandler
from fastllm.cli.utils import get_org, get_project

from fastllm.apis.base import APIClient
import fastllm.utils.logger as logger
from fastllm.utils.config_utils import read_config, upsert_config
from fastllm.fastllm import FastLLM
from fastllm.utils.crypto import generate_api_key, encrypt_message
from fastllm.llms.llm_dev import LLMDev
from fastllm.database.crud import (
    create_llm_module_version,
    create_prompt,
    create_run_log,
    list_llm_modules,
    list_llm_module_versions,
    list_prompts,
    list_run_logs,
    get_sample_input,
    get_llm_module_uuid,
    update_local_usage_llm_module_by_name,
    create_llm_modules,
    create_llm_module_versions,
    create_prompts,
    create_run_logs,
    update_samples
)
from fastllm.database.models import (
    LLMModule,
    LLMModuleVersion,
    Prompt,
    RunLog,
    SampleInputs,
)
from fastllm.utils.enums import (
    ServerTask,
    LocalTask,
    LLMModuleVersionStatus,
    ChangeLogAction,
)

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
    def __init__(self, fastllm_client: FastLLM):
        self.fastllm_client: FastLLM = fastllm_client
        self.rwlock = rwlock.RWLockFair()

    async def _get_llm_modules(self, llm_module_name: str):
        """Get llm_module from registry"""
        with self.rwlock.gen_rlock():
            llm_module = next(
                (
                    llm_module
                    for llm_module in self.fastllm_client.llm_modules
                    if llm_module.name == llm_module_name
                ),
                None,
            )
        return llm_module

    def update_client_instance(self, new_client):
        with self.rwlock.gen_wlock():
            self.fastllm_client = new_client
            if self.ws is not None:
                asyncio.run(self.ws.send(json.dumps({"type" : ServerTask.LOCAL_UPDATE_ALERT.value})))

    async def __handle_message(
        self, message: Dict[str, Any], ws: WebSocketClientProtocol
    ):
        logger.info(f"Received message: {message}")
        response: Dict[Any, str] = {}
        # If the message has a correlation_id, add it to the response
        # correlation_id is the unique ID of the function from backend to local
        if message.get("correlation_id"):
            response["correlation_id"] = message["correlation_id"]

        # If the message has a runner_id, add it to the response
        if message.get("runner_id"):
            response["runner_id"] = message["runner_id"]

        try:
            if message["type"] == LocalTask.LIST_MODULES:
                res_from_local_db = list_llm_modules()
                modules_with_local_usage = [module for module in res_from_local_db if module['local_usage'] == True]
                data = {"llm_modules": modules_with_local_usage}

            elif message["type"] == LocalTask.LIST_VERSIONS:
                llm_module_uuid = message["llm_module_uuid"]
                res_from_local_db = list_llm_module_versions(llm_module_uuid)
                data = {"llm_module_versions": res_from_local_db}

            elif message["type"] == LocalTask.GET_PROMPTS:
                llm_module_version_uuid = message["llm_module_version_uuid"]
                res_from_local_db = list_prompts(llm_module_version_uuid)
                data = {"prompts": res_from_local_db}

            elif message["type"] == LocalTask.GET_RUN_LOGS:
                llm_module_version_uuid = message["llm_module_version_uuid"]
                res_from_local_db = list_run_logs(llm_module_version_uuid)
                data = {"run_logs": res_from_local_db}

            elif message["type"] == LocalTask.RUN_LLM_MODULE:
                llm_module_name = message["llm_module_name"]
                sample_name = message["sample_name"]
                # get sample from db
                if sample_name is not None:
                    sample_input_row = get_sample_input(sample_name)
                    if sample_input_row is None or 'contents' not in sample_input_row:
                        logger.error(
                                f"There is no sample input {sample_name}."
                            )
                        return
                    sample_input = sample_input_row['contents']
                else:
                    sample_input = None
                
                llm_module_names = [llm_module.name for llm_module in self.fastllm_client.llm_modules]
                if llm_module_name not in llm_module_names:
                    logger.error(
                            f"There is no llm_module {llm_module_name}."
                        )
                    return
                
                try:
                    logger.info(f"Started service: {llm_module_name}")
                    # llm_module = await self._get_llm_modules(llm_module_name)
            
                    # create llm_module_dev_instance
                    llm_module_dev = LLMDev()
                    # fine llm_module_uuid from local db
                    llm_module_uuid = get_llm_module_uuid(llm_module_name)['uuid']
                
                    llm_module_version_uuid = message['llm_module_version_uuid']
                    # If llm_module_version_uuid is None, create new version & prompt
                    if llm_module_version_uuid is None:
                        llm_module_version: LLMModuleVersion = create_llm_module_version(
                            llm_module_uuid=llm_module_uuid,
                            status=LLMModuleVersionStatus.BROKEN.value,
                            from_uuid=message['from_uuid']
                        )
                        llm_module_version_uuid = llm_module_version.uuid

                        prompts = message["prompts"]
                        for prompt in prompts:
                            create_prompt(
                                version_uuid=llm_module_version_uuid,
                                type=prompt["type"],
                                step=prompt["step"],
                                content=prompt["content"],
                            )
                        # send message to backend
                        data = {
                            "type": ServerTask.UPDATE_RESULT_RUN.value,
                            "llm_module_version_uuid": llm_module_version_uuid,
                            "status": "running",
                        }
                        await ws.send(json.dumps(data))
                    
                    model = message['model']
                    prompts = message['prompts']
                    parsing_type = message['parsing_type']
                    
                    if sample_input is not None:
                        messages_for_run = [{'content': prompt['content'].format(**sample_input), 'role': prompt['role']} for prompt in prompts]

                    else:
                        messages_for_run = prompts
                    output = {
                        "raw_output" : "",
                        "parsed_outputs" : {}
                    }
                    async for item in llm_module_dev.dev_generate(messages_for_run, parsing_type, model):
                        # send item to backend
                        # save item & parse
                        # if type(item) == str: raw output, if type(item) == dict: parsed output
                        if type(item) == str:
                            output['raw_output'] += item
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "raw_output": item
                            }
                        elif type(item) == dict:
                            if list(item.keys())[0] not in output['parsed_outputs']:
                                output['parsed_outputs'][list(item.keys())[0]] = list(item.values())[0]
                            else:
                                output['parsed_outputs'][list(item.keys())[0]] += list(item.values())[0]
                            data = {
                                "type": ServerTask.UPDATE_RESULT_RUN.value,
                                "status": "running",
                                "parsed_outputs": item
                            }
                        await ws.send(json.dumps(data))

                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "raw_output": output["raw_output"],
                        "parsed_outputs": output["parsed_outputs"],
                        "status": "completed",
                    }
                except Exception as error:
                    data = {
                        "type": ServerTask.UPDATE_RESULT_RUN.value,
                        "status": "failed",
                        "log": str(error),
                    }
                create_run_log(
                    llm_module_version_uuid=llm_module_version_uuid,
                    inputs=json.dumps(sample_input),
                    raw_output=json.dumps(output["raw_output"]),
                    parsed_outputs=json.dumps(output["parsed_outputs"]),
                )
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
                        "Connected to gateway. Your FastLLM Dev is now online! 🎉"
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


class CodeReloadHandler(FileSystemEventHandler):
    def __init__(
        self,
        fastllm_client_filename: str,
        fastllm_variable_name: str,
        dev_websocket_client: DevWebsocketClient,
    ):
        self.fastllm_client_filename: str = fastllm_client_filename
        self.client_variable_name: str = fastllm_variable_name
        self.dev_websocket_client: DevWebsocketClient = dev_websocket_client  # 저장
        self.timer = None

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if event.src_path.endswith(".py"):
            if self.timer is not None:
                self.timer.cancel()
            # reload modified file & main file
            self.timer = Timer(0.5, self.reload_code, args=(event.src_path,))
            self.timer.start()

    def reload_code(self, modified_file_path: str):
        print(f"[violet]fastllm:dev:[/violet]  Reloading {self.fastllm_client_filename} module due to changes...")
        # Reload the client module
        module_name = modified_file_path.replace("./", "").replace("/", ".")[
            :-3
        ]  # assuming the file is in the PYTHONPATH
        if module_name in sys.modules:
            # print(module_name)
            module = sys.modules[module_name]
            importlib.reload(module)

        reloaded_module = importlib.reload(sys.modules[self.fastllm_client_filename])
        print(f"[violet]fastllm:dev:[/violet]  {self.fastllm_client_filename} module reloaded successfully.")

        new_client_instance: FastLLM = getattr(
            reloaded_module, self.client_variable_name
        )
        # print(new_client_instance.llm_modules)
        new_llm_module_name_list = [
            llm_module.name for llm_module in new_client_instance.llm_modules
        ]
        old_llm_module_name_list = [
            llm_module.name
            for llm_module in self.dev_websocket_client.fastllm_client.llm_modules
        ]

        # 사라진 llm_modules 에 대해 local db llm_module.local_usage False Update
        removed_name_list = list(
            set(old_llm_module_name_list) - set(new_llm_module_name_list)
        )
        for removed_name in removed_name_list:
            update_local_usage_llm_module_by_name(removed_name, False)

        # 새로 생긴 llm_module 에 대해 local db llm_module.local_usage True Update
        # TODO: 좀 더 specific 한 API와 연결 필요
        config = read_config()
        org = config["dev_branch"]["org"]
        project = config["dev_branch"]["project"]
        project_status = APIClient.execute(
            method="GET",
            path="/pull_project",
            params={"project_uuid": project["uuid"]},
        ).json()

        changelogs = APIClient.execute(
            method="GET",
            path="/get_changelog",
            params={
                "project_uuid": project["uuid"],
                "local_project_version": config["dev_branch"]["project_version"],
                "levels": [1, 2],
            },
        ).json()
        # IF local_usage=False 인 name=name 이 있을 경우, local_usage=True
        update_by_changelog_for_reload(
            changelogs=changelogs,
            project_status=project_status,
            local_code_llm_module_name_list=new_llm_module_name_list,
        )

        for llm_module in new_client_instance.llm_modules:
            if llm_module.name not in old_llm_module_name_list:
                update_local_usage_llm_module_by_name(llm_module.name, True)
                
        # update samples in local DB
        update_samples(new_client_instance.samples)
        self.dev_websocket_client.update_client_instance(new_client_instance)


def update_by_changelog_for_reload(
    changelogs: list[dict],
    project_status: dict,
    local_code_llm_module_name_list: list[str],
):
    """Update Local DB by changelog"""
    local_db_llm_module_list: list = list_llm_modules()  # {"name", "uuid"}

    for changelog in changelogs:
        level: int = changelog["level"]
        action: str = changelog["changelog"]["action"]
        # table:str = changelog['changelog']['object']
        uuid_list: list = changelog["changelog"]["identifiers"]
        if level == 1:
            if action == ChangeLogAction.ADD.value:
                llm_module_list = [
                    x for x in project_status["llm_modules"] if x["uuid"] in uuid_list
                ]
                for llm_module in llm_module_list:
                    local_db_llm_module_name_list = [
                        x["name"] for x in local_db_llm_module_list
                    ]

                    if llm_module["name"] not in local_db_llm_module_name_list:
                        # IF llm_module not in Local DB
                        if llm_module["name"] in local_code_llm_module_name_list:
                            # IF llm_module in Local Code
                            llm_module["local_usage"] = True
                            llm_module["is_deployment"] = True
                            create_llm_modules([llm_module])
                        else:
                            llm_module["local_usage"] = False
                            llm_module["is_deployment"] = True
                            create_llm_modules([llm_module])
            else:
                # TODO: add code DELETE, CHANGE, FIX later
                pass
            previous_version_levels = changelog["previous_version"].split(".")
            current_version_levels = [
                str(int(previous_version_levels[0]) + 1),
                "0",
                "0",
            ]
            current_version = ".".join(current_version_levels)
        elif level == 2:
            if action == ChangeLogAction.ADD.value:
                # find llm_module_version in project_status['llm_module_versions'] where uuid in uuid_list
                llm_module_version_list = [
                    x
                    for x in project_status["llm_module_versions"]
                    if x["uuid"] in uuid_list
                ]
                # check if llm_module_version['name'] is in local_code_llm_module_list
                llm_module_version_list_to_update = [
                    x
                    for x in llm_module_version_list
                    if x["name"] in local_code_llm_module_name_list
                ]
                update_uuid_list = [
                    x["uuid"] for x in llm_module_version_list_to_update
                ]

                # find prompts and run_logs to update
                prompts_to_update = [
                    x
                    for x in project_status["prompts"]
                    if x["version_uuid"] in update_uuid_list
                ]
                run_logs_to_update = [
                    x
                    for x in project_status["run_logs"]
                    if x["version_uuid"] in update_uuid_list
                ]

                for llm_module_version in llm_module_version_list_to_update:
                    llm_module_version["candidate_version"] = llm_module_version[
                        "version"
                    ]
                    del llm_module_version["version"]
                    llm_module_version[
                        "status"
                    ] = LLMModuleVersionStatus.CANDIDATE.value

                create_llm_module_versions(llm_module_version_list_to_update)
                create_prompts(prompts_to_update)
                create_run_logs(run_logs_to_update)

                local_db_llm_module_list += [
                    {"name": x["name"], "uuid": x["uuid"]}
                    for x in llm_module_version_list_to_update
                ]
            else:
                pass
            previous_version_levels = changelog["previous_version"].split(".")
            current_version_levels = [
                previous_version_levels[0],
                str(int(previous_version_levels[1]) + 1),
                "0",
            ]
            current_version = ".".join(current_version_levels)
        else:
            pass
            previous_version_levels = changelog["previous_version"].split(".")
            current_version_levels = [
                previous_version_levels[0],
                previous_version_levels[1],
                str(int(previous_version_levels[2]) + 1),
            ]
            current_version = ".".join(current_version_levels)

        upsert_config({"project_version": current_version}, section="dev_branch")
    return True
