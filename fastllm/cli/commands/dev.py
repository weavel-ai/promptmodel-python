import os
import time
import asyncio
import typer
import json
import sys
import importlib
import signal
from importlib import resources
from requests import request
from typing import Callable, Dict, Any, List

import webbrowser
import inspect

from rich import print
from InquirerPy import inquirer
from watchdog.observers import Observer

from fastllm.fastllm import FastLLM
from fastllm.apis.base import APIClient
from fastllm.constants import ENDPOINT_URL, WEB_CLIENT_URL
from fastllm.cli.utils import get_org, get_project
from fastllm.cli.signal_handler import dev_terminate_signal_handler
import fastllm.utils.logger as logger
from fastllm.utils.config_utils import read_config, upsert_config
from fastllm.utils.crypto import generate_api_key, encrypt_message
from fastllm.utils.enums import LLMModuleVersionStatus, ChangeLogAction
from fastllm.websocket.clients import DevWebsocketClient, CodeReloadHandler
from fastllm.database.orm import initialize_db
from fastllm.database.crud import (
    create_llm_modules,
    create_llm_module_versions,
    create_prompts,
    create_run_logs,
    list_llm_modules,
    create_llm_module,
    create_llm_module_version,
    create_prompt,
    create_run_log,
    update_is_deployment_llm_module,
    hide_llm_module_not_in_code,
    update_samples
)

FASTLLM_DEV_FILENAME = os.path.join(os.getcwd(), "fastllm_dev.py")
FASTLLM_DEV_STARTER_FILENAME = "STARTER.py"

def dev():
    """Creates a new prompt development environment, and opens up FastLLM in the browser."""
    upsert_config({"initializing" : True}, "dev_branch")
    signal.signal(signal.SIGINT, dev_terminate_signal_handler)
    import os

    if not os.path.exists(FASTLLM_DEV_FILENAME):
        # Read the content from the source file
        content = resources.read_text("fastllm", "STARTER.py")

        # Write the content to the target file
        with open(FASTLLM_DEV_FILENAME, "w") as target_file:
            target_file.write(content)

    fastllm_client_filename, client_variable_name = "fastllm_dev:app".split(":")
    
    # Init local database & open
    initialize_db()
    
    config = read_config()
    
    client_module = importlib.import_module(fastllm_client_filename)
    client_instance: FastLLM = getattr(client_module, client_variable_name)

    
    if "name" in config["dev_branch"]:
        org = get_org(config)
        project = get_project(config=config, org=org)
        validate_branch_name = lambda name: APIClient.execute(
            method="GET", path="/check_dev_branch_name", params={"name": name}
        ).json()
        branch_name = inquirer.text(
            message="Enter a development branch name:",
            validate=lambda x: " " not in x and validate_branch_name(x),
            invalid_message="Branch name already exists or contains spaces.",
        ).execute()
        
        upsert_config({"name": branch_name, "project" : project, "org": org}, section="dev_branch")

        print("\nCreating local development branch...")
        APIClient.execute(
            method="POST",
            path="/create_dev_branch",
            params={"name": branch_name, "project_uuid": project["uuid"]},
        )      
        
        # connect
        res = APIClient.execute(
            method="POST",
            path="/connect_cli_dev",
            params={"project_uuid": project["uuid"], "branch_name": branch_name},
        )
        if res.status_code != 200:
            print(f"Error: {res.json()['detail']}")
            return
        
        # fetch current project status
        project_status = APIClient.execute(
            method="GET",
            path="/pull_project",
            params={"project_uuid": project["uuid"]},
        ).json()
        # save project version
        upsert_config({"project_version" : project_status["project_version"]}, section="dev_branch")
        
        local_llm_modules = client_instance.llm_modules
        local_llm_module_names = [x.name for x in local_llm_modules]
        
        # save llm_modules
        for llm_module in project_status["llm_modules"]:
            llm_module['is_deployment'] = True
            if llm_module['name'] in local_llm_module_names:
                llm_module['local_usage'] = True
            else:
                llm_module['local_usage'] = False
                
        create_llm_modules(project_status["llm_modules"])
        
        # save llm_module_versions
        for version in project_status['llm_module_versions']:
            version['candidate_version'] = version['version']
            del version['version']
            version['status'] = LLMModuleVersionStatus.CANDIDATE.value
        create_llm_module_versions(project_status["llm_module_versions"])
        # save prompts
        create_prompts(project_status["prompts"])
        # save run_logs
        create_run_logs(project_status["run_logs"])
        
        # create llm_modules from code in local DB
        project_llm_module_names = [x['name'] for x in project_status["llm_modules"]]
        only_in_local = list(set(local_llm_module_names) - set(project_llm_module_names))
        only_in_local_llm_modules = [{"name" : x, "project_uuid" : project['uuid']} for x in only_in_local]
        create_llm_modules(only_in_local_llm_modules)
        
    else:
        org = config["dev_branch"]["org"]
        project = config["dev_branch"]["project"]
        branch_name = config["dev_branch"]["name"]
        
        res = APIClient.execute(
            method="POST",
            path="/connect_cli_dev",
            params={"project_uuid": project["uuid"], "branch_name": branch_name},
        )
        if res.status_code != 200:
            print(f"Error: {res.json()['detail']}")
            return
        
        # Get changelog & update DB
        # fetch current project status
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
                "levels" : [1,2]
            },
        ).json()
        local_code_llm_module_name_list = [x.name for x in client_instance.llm_modules]
        res = update_by_changelog(changelogs, project_status, local_code_llm_module_name_list)
        if res is False:
            print("Update Dev Stopped.")
            upsert_config({"online": False}, section="dev_branch")
            return
        
        # local_code_llm_module_name_list 에 없는 llm_module 의 local_usage=False 설정
        hide_llm_module_not_in_code(local_code_llm_module_name_list)
        
        # 새로 생긴 llm_module DB에 생성
        local_db_llm_module_names = [x['name'] for x in list_llm_modules()]
        only_in_local = list(set(local_code_llm_module_name_list) - set(local_db_llm_module_names))
        only_in_local_llm_modules = [{"name" : x, "project_uuid" : project['uuid']} for x in only_in_local]
        create_llm_modules(only_in_local_llm_modules)
    
    dev_url = f"{WEB_CLIENT_URL}/org/{org['slug']}/project/{project['uuid']}/dev/{branch_name}"
    
    # Open websocket connection to backend server
    dev_websocket_client = DevWebsocketClient(fastllm_client=client_instance)
    
    import threading
    reloader_thread = threading.Thread(
        target=start_code_reloader, 
        args=(
            fastllm_client_filename,
            client_variable_name,
            dev_websocket_client
            )
        )
    reloader_thread.daemon = True  # Set the thread as a daemon
    reloader_thread.start()
    
    print(
        f"\nOpening [violet]FastLLM[/violet] prompt engineering environment with the following configuration:\n"
    )
    print(f"📌 Organization: [blue]{org['name']}[/blue]")
    print(f"📌 Project: [blue]{project['name']}[/blue]")
    print(f"📌 Local development branch: [green][bold]{branch_name}[/bold][/green]")
    print(
        f"\nIf browser doesn't open automatically, please visit [link={dev_url}]{dev_url}[/link]"
    )
    webbrowser.open(dev_url)
    
    upsert_config({"online": True, "initializing": False}, section="dev_branch")
    # save samples to local DB
    update_samples(client_instance.samples)
    
    # 웹소켓 연결 열기
    asyncio.run(
        dev_websocket_client.connect_to_gateway(
            project_uuid=project["uuid"],
            dev_branch_name=branch_name,
            cli_access_header=APIClient._get_headers()
        )
    )

app = typer.Typer(invoke_without_command=True, callback=dev)

def start_code_reloader(
    fastllm_client_filename,
    client_variable_name,
    dev_websocket_client
):
    event_handler = CodeReloadHandler(
    fastllm_client_filename,
    client_variable_name,
    dev_websocket_client
    )
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


    
def update_by_changelog(
    changelogs: list[dict],
    project_status: dict,
    local_code_llm_module_name_list: list[str]
):
    """Update Local DB by changelog"""
    local_db_llm_module_list: list = list_llm_modules() # {"name", "uuid"}
    
    for changelog in changelogs:
        level:int = changelog['level']
        action:str = changelog['changelog']['action']
        # table:str = changelog['changelog']['object']
        uuid_list:list = changelog['changelog']['identifiers']
        if level == 1:
            if action == ChangeLogAction.ADD.value:
                llm_module_list = [x for x in project_status['llm_modules'] if x['uuid'] in uuid_list]
                for llm_module in llm_module_list:
                    local_db_llm_module_name_list = [x['name'] for x in local_db_llm_module_list]
                    
                    if llm_module['name'] not in local_db_llm_module_name_list:
                        # IF llm_module not in Local DB
                        if llm_module['name'] in local_code_llm_module_name_list:
                            # IF llm_module in Local Code
                            llm_module['local_usage'] = True
                            llm_module['is_deployment'] = True
                            create_llm_modules([llm_module])
                        else:
                            llm_module['local_usage'] = False
                            llm_module['is_deployment'] = True
                            create_llm_modules([llm_module])
                    else:
                        local_db_llm_module = [x for x in local_db_llm_module_list if x['name'] == llm_module['name']][0]
                        if local_db_llm_module['is_deployment'] is False:
                            print("Creation of llm_module with identical name was detected in local & deployment.")
                            check_same = inquirer.confirm(
                                message="Are they same llm_module? [y/n]",
                                default=False
                            ).execute()
                            if not check_same:
                                print(f"Please rename local llm_module {llm_module['name']} to continue")
                                return False
                            update_is_deployment_llm_module(local_db_llm_module['uuid'], is_deployment=True)
            else:
                # TODO: add code DELETE, CHANGE, FIX later
                pass
            previous_version_levels = changelog['previous_version'].split(".")
            current_version_levels = [str(int(previous_version_levels[0]) + 1), "0", "0"]
            current_version = ".".join(current_version_levels)
        elif level == 2:
            if action == ChangeLogAction.ADD.value:            
                # find llm_module_version in project_status['llm_module_versions'] where uuid in uuid_list
                llm_module_version_list = [x for x in project_status['llm_module_versions'] if x['uuid'] in uuid_list]
                # check if llm_module_version['name'] is in local_code_llm_module_list
                llm_module_version_list_to_update = [x for x in llm_module_version_list if x['name'] in local_code_llm_module_name_list]
                update_uuid_list = [x['uuid'] for x in llm_module_version_list_to_update]
                
                # find prompts and run_logs to update
                prompts_to_update = [x for x in project_status['prompts'] if x['version_uuid'] in update_uuid_list]
                run_logs_to_update = [x for x in project_status['run_logs'] if x['version_uuid'] in update_uuid_list]
                
                for llm_module_version in llm_module_version_list_to_update:
                    llm_module_version['candidate_version'] = llm_module_version['version']
                    del llm_module_version['version']
                    llm_module_version['status'] = LLMModuleVersionStatus.CANDIDATE.value
                    
                create_llm_module_versions(llm_module_version_list_to_update)
                create_prompts(prompts_to_update)
                create_run_logs(run_logs_to_update)
                
                local_db_llm_module_list += [{"name" : x['name'], "uuid" : x['uuid']} for x in llm_module_version_list_to_update]
            else:
                pass
            previous_version_levels = changelog['previous_version'].split(".")
            current_version_levels = [previous_version_levels[0], str(int(previous_version_levels[1]) + 1), "0"]
            current_version = ".".join(current_version_levels)
        else:
            pass
            previous_version_levels = changelog['previous_version'].split(".")
            current_version_levels = [previous_version_levels[0], previous_version_levels[1], str(int(previous_version_levels[2]) + 1)]
            current_version = ".".join(current_version_levels)
    
        upsert_config({"project_version" : current_version}, section="dev_branch")
    return True