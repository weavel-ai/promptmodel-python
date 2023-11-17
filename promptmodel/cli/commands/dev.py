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
from playhouse.shortcuts import model_to_dict


import webbrowser
import inspect

from rich import print
from InquirerPy import inquirer
from watchdog.observers import Observer

from promptmodel import DevApp
from promptmodel.apis.base import APIClient
from promptmodel.constants import ENDPOINT_URL, WEB_CLIENT_URL
from promptmodel.cli.commands.init import init as promptmodel_init
from promptmodel.cli.utils import get_org, get_project
from promptmodel.cli.signal_handler import dev_terminate_signal_handler
from promptmodel.utils import logger
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils.crypto import generate_api_key, encrypt_message
from promptmodel.types.enums import ModelVersionStatus, ChangeLogAction
from promptmodel.websocket import DevWebsocketClient, CodeReloadHandler
from promptmodel.database.orm import initialize_db
from promptmodel.database.models import *
from promptmodel.database.crud import (
    hide_prompt_model_not_in_code,
    hide_chat_model_not_in_code,
    update_samples,
    update_prompt_model_uuid,
    update_chat_model_uuid,
    rename_prompt_model,
    rename_chat_model,
)


def dev():
    """Creates a new prompt development environment, and opens up DevApp in the browser."""
    upsert_config({"initializing": True}, "dev_branch")
    signal.signal(signal.SIGINT, dev_terminate_signal_handler)
    promptmodel_init(from_cli=False)

    _devapp_filename, devapp_instance_name = "promptmodel_dev:app".split(":")

    # Init local database & open
    initialize_db()

    config = read_config()

    devapp_module = importlib.import_module(_devapp_filename)
    devapp_instance: DevApp = getattr(devapp_module, devapp_instance_name)

    if "name" not in config["dev_branch"]:
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

        print("\nCreating local development branch...")
        created_dev_branch_row = APIClient.execute(
            method="POST",
            path="/create_dev_branch",
            params={"name": branch_name, "project_uuid": project["uuid"]},
        )
        upsert_config(
            {
                "name": branch_name,
                "project": project,
                "org": org,
                "uuid": created_dev_branch_row.json()["uuid"],
                "project_version": "0.0.0",
            },
            section="dev_branch",
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
        upsert_config(
            {"project_version": project_status["project_version"]}, section="dev_branch"
        )

        local_prompt_model_names = devapp_instance._get_prompt_model_name_list()
        local_chat_model_names = devapp_instance._get_chat_model_name_list()

        # save prompt_models
        for prompt_model in project_status["prompt_models"]:
            prompt_model["is_deployed"] = True
            if prompt_model["name"] in local_prompt_model_names:
                prompt_model["used_in_code"] = True
            else:
                prompt_model["used_in_code"] = False

        PromptModel.insert_many(project_status["prompt_models"]).execute()

        # save prompt_model_versions
        for version in project_status["prompt_model_versions"]:
            version["status"] = ModelVersionStatus.CANDIDATE.value
        PromptModelVersion.insert_many(
            project_status["prompt_model_versions"]
        ).execute()
        # save prompts
        Prompt.insert_many(project_status["prompts"]).execute()
        # save run_logs
        RunLog.insert_many(project_status["run_logs"]).execute()

        # create prompt_models from code in local DB
        project_prompt_model_names = [
            x["name"] for x in project_status["prompt_models"]
        ]
        only_in_local = list(
            set(local_prompt_model_names) - set(project_prompt_model_names)
        )
        only_in_local_prompt_models = [
            {"name": x, "project_uuid": project["uuid"]} for x in only_in_local
        ]
        PromptModel.insert_many(only_in_local_prompt_models).execute()

        # save chat_models
        for chat_model in project_status["chat_models"]:
            chat_model["is_deployed"] = True
            if chat_model["name"] in local_chat_model_names:
                chat_model["used_in_code"] = True
            else:
                chat_model["used_in_code"] = False

        ChatModel.insert_many(project_status["chat_models"]).execute()

        # save chat_model_versions
        for version in project_status["chat_model_versions"]:
            version["status"] = ModelVersionStatus.CANDIDATE.value
        ChatModelVersion.insert_many(project_status["chat_model_versions"]).execute()

        # create_chat_models from code in local DB
        project_chat_model_names = [x["name"] for x in project_status["chat_models"]]
        only_in_local_names = list(
            set(local_chat_model_names) - set(project_chat_model_names)
        )
        only_in_local_chat_models = [
            {"name": x, "project_uuid": project["uuid"]} for x in only_in_local_names
        ]
        ChatModel.insert_many(only_in_local_chat_models).execute()

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
                "levels": [1, 2],
            },
        ).json()
        local_code_prompt_model_name_list = (
            devapp_instance._get_prompt_model_name_list()
        )
        local_code_chat_model_name_list = devapp_instance._get_chat_model_name_list()

        res = update_by_changelog(
            changelogs,
            project_status,
            local_code_prompt_model_name_list,
            local_code_chat_model_name_list,
        )

        if res is False:
            print("Update Dev Stopped.")
            upsert_config({"online": False}, section="dev_branch")
            return

        # Make prompt_model.used_in_code=False which is not in local code
        hide_prompt_model_not_in_code(local_code_prompt_model_name_list)
        # Make chat_model.used_in_code=False which is not in local code
        hide_chat_model_not_in_code(local_code_chat_model_name_list)

        # Save new prompt_model in code to local DB
        local_db_prompt_model_names = [
            x["name"]
            for x in [model_to_dict(x, recurse=False) for x in [PromptModel.select()]]
        ]
        only_in_local = list(
            set(local_code_prompt_model_name_list) - set(local_db_prompt_model_names)
        )
        only_in_local_prompt_models = [
            {"name": x, "project_uuid": project["uuid"]} for x in only_in_local
        ]
        PromptModel.insert_many(only_in_local_prompt_models).execute()

        # Save new chat_model in code to local DB
        local_db_chat_model_names = [
            x["name"]
            for x in [model_to_dict(x, recurse=False) for x in [ChatModel.select()]]
        ]
        only_in_local = list(
            set(local_code_chat_model_name_list) - set(local_db_chat_model_names)
        )
        only_in_local_chat_models = [
            {"name": x, "project_uuid": project["uuid"]} for x in only_in_local
        ]
        ChatModel.insert_many(only_in_local_chat_models).execute()

    dev_url = f"{WEB_CLIENT_URL}/org/{org['slug']}/projects/{project['uuid']}/dev/{branch_name}"

    # Open websocket connection to backend server
    dev_websocket_client = DevWebsocketClient(_devapp=devapp_instance)

    import threading

    reloader_thread = threading.Thread(
        target=start_code_reloader,
        args=(_devapp_filename, devapp_instance_name, dev_websocket_client),
    )
    reloader_thread.daemon = True  # Set the thread as a daemon
    reloader_thread.start()

    print(
        f"\nOpening [violet]Promptmodel[/violet] prompt engineering environment with the following configuration:\n"
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
    update_samples(devapp_instance.samples)

    # Open Websocket
    asyncio.run(
        dev_websocket_client.connect_to_gateway(
            project_uuid=project["uuid"],
            dev_branch_name=branch_name,
            cli_access_header=APIClient._get_headers(),
        )
    )


app = typer.Typer(invoke_without_command=True, callback=dev)


def start_code_reloader(_devapp_filename, devapp_instance_name, dev_websocket_client):
    event_handler = CodeReloadHandler(
        _devapp_filename, devapp_instance_name, dev_websocket_client
    )
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def update_by_changelog(
    changelogs: List[Dict],
    project_status: dict,
    local_code_prompt_model_name_list: List[str],
    local_code_chat_model_name_list: List[str],
):
    """Update Local DB by changelog"""
    local_db_prompt_model_list: List = [
        model_to_dict(x, recurse=False) for x in [PromptModel.select()]
    ]  # {"name", "uuid"}
    local_db_chat_model_list: List = [
        model_to_dict(x, recurse=False) for x in [ChatModel.select()]
    ]  # {"name", "uuid"}

    for changelog in changelogs:
        level: int = changelog["level"]
        logs = changelog["logs"]
        if level == 1:
            for log in logs:
                subject = log["subject"]
                action: str = log["action"]
                if subject == "prompt_model":
                    (
                        is_success,
                        local_db_prompt_model_list,
                    ) = update_prompt_model_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_prompt_model_list=local_db_prompt_model_list,
                        local_code_prompt_model_name_list=local_code_prompt_model_name_list,
                    )

                    if not is_success:
                        return False

                elif subject == "prompt_model_version":
                    local_db_prompt_model_list = update_prompt_model_version_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_prompt_model_list=local_db_prompt_model_list,
                        local_code_prompt_model_name_list=local_code_prompt_model_name_list,
                    )

                elif subject == "chat_model":
                    is_success, local_db_chat_model_list = update_chat_model_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_chat_model_list=local_db_chat_model_list,
                        local_code_chat_model_name_list=local_code_chat_model_name_list,
                    )

                    if not is_success:
                        return False

                elif subject == "chat_model_version":
                    local_db_chat_model_list = update_chat_model_version_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_chat_model_list=local_db_chat_model_list,
                        local_code_chat_model_name_list=local_code_chat_model_name_list,
                    )

                else:
                    pass
            previous_version_levels = changelog["previous_version"].split(".")
            current_version_levels = [
                str(int(previous_version_levels[0]) + 1),
                "0",
                "0",
            ]
            current_version = ".".join(current_version_levels)
        elif level == 2:
            for log in logs:
                subject = log["subject"]
                action: str = log["action"]
                uuid_list: list = log["identifiers"]
                if subject == "prompt_model_version":
                    local_db_prompt_model_list = update_prompt_model_version_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_prompt_model_list=local_db_prompt_model_list,
                        local_code_prompt_model_name_list=local_code_prompt_model_name_list,
                    )

                elif subject == "chat_model_version":
                    local_db_chat_model_list = update_chat_model_version_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_chat_model_list=local_db_chat_model_list,
                        local_code_chat_model_name_list=local_code_chat_model_name_list,
                    )
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
            previous_version_levels = changelog["previous_version"].split(".")
            current_version_levels = [
                previous_version_levels[0],
                previous_version_levels[1],
                str(int(previous_version_levels[2]) + 1),
            ]
            current_version = ".".join(current_version_levels)

        upsert_config({"project_version": current_version}, section="dev_branch")
    return True


def update_prompt_model_changelog(
    action: ChangeLogAction,
    project_status: dict,
    uuid_list: List[str],
    local_db_prompt_model_list: List[Dict],
    local_code_prompt_model_name_list: List[str],
):
    if action == ChangeLogAction.ADD.value:
        prompt_model_list = [
            x for x in project_status["prompt_models"] if x["uuid"] in uuid_list
        ]
        for prompt_model in prompt_model_list:
            local_db_prompt_model_name_list = [
                x["name"] for x in local_db_prompt_model_list
            ]

            if prompt_model["name"] not in local_db_prompt_model_name_list:
                # IF prompt_model not in Local DB
                if prompt_model["name"] in local_code_prompt_model_name_list:
                    # IF prompt_model in Local Code
                    prompt_model["used_in_code"] = True
                    prompt_model["is_deployed"] = True
                else:
                    prompt_model["used_in_code"] = False
                    prompt_model["is_deployed"] = True
                PromptModel.create(**prompt_model)
            else:
                local_db_prompt_model = [
                    x
                    for x in local_db_prompt_model_list
                    if x["name"] == prompt_model["name"]
                ][0]
                if local_db_prompt_model["is_deployed"] is False:
                    print(
                        "Creation of promptmodel with identical name was detected in local & deployment."
                    )
                    check_same = inquirer.confirm(
                        message="Are they same promptmodel? [y/n]", default=False
                    ).execute()
                    # check_same = input()
                    # if check_same == "y":
                    #     check_same = True
                    # else:
                    #     check_same = False

                    if not check_same:
                        # rename & change name in Local DB
                        print(
                            f"Please rename local promptmodel {prompt_model['name']} to continue"
                        )
                        validate_new_promptmodel_name = (
                            lambda name: name not in local_code_prompt_model_name_list
                            and name not in local_db_prompt_model_name_list
                        )
                        new_model_name = inquirer.text(
                            message="Enter the new promptmodel name:",
                            validate=lambda x: validate_new_promptmodel_name(x),
                            invalid_message="promptmodel name already exists.",
                        ).execute()
                        # new_model_name = input()
                        rename_prompt_model(
                            local_db_prompt_model["uuid"], new_model_name
                        )

                        print("We changed the name of promptmodel in local DB.")
                        print(
                            f"Please change the name of promptmodel '{prompt_model['name']}' in your project code and restart."
                        )
                        # dev 꺼버리고, 수정하고 다시 키라고 명령.
                        return False, local_db_prompt_model_list
                    update_prompt_model_uuid(
                        local_db_prompt_model["uuid"], prompt_model["uuid"]
                    )
                    local_db_prompt_model_list: list = [
                        model_to_dict(x, recurse=False) for x in [PromptModel.select()]
                    ]
    else:
        # TODO: add code DELETE, CHANGE, FIX later
        pass
    return True, local_db_prompt_model_list


def update_prompt_model_version_changelog(
    action: ChangeLogAction,
    project_status: dict,
    uuid_list: List[str],
    local_db_prompt_model_list: List[Dict],
    local_code_prompt_model_name_list: List[str],
) -> List[Dict[str, Any]]:
    if action == ChangeLogAction.ADD.value:
        # find prompt_model_version in project_status['prompt_model_versions'] where uuid in uuid_list
        prompt_model_version_list_in_changelog = [
            x for x in project_status["prompt_model_versions"] if x["uuid"] in uuid_list
        ]

        # check if prompt_model_version['uuid'] is in local_db_prompt_model_list
        local_db_version_uuid_list = [
            str(x.uuid) for x in list(PromptModelVersion.select())
        ]
        version_list_to_update = [
            x
            for x in prompt_model_version_list_in_changelog
            if x["uuid"] not in local_db_version_uuid_list
        ]
        version_uuid_list_to_update = [x["uuid"] for x in version_list_to_update]

        # find prompts and run_logs to update
        prompts_to_update = [
            x
            for x in project_status["prompts"]
            if x["version_uuid"] in version_uuid_list_to_update
        ]
        run_logs_to_update = [
            x
            for x in project_status["run_logs"]
            if x["version_uuid"] in version_uuid_list_to_update
        ]

        for prompt_model_version in version_list_to_update:
            prompt_model_version["status"] = ModelVersionStatus.CANDIDATE.value

        PromptModelVersion.insert_many(version_list_to_update).execute()
        Prompt.insert_many(prompts_to_update).execute()
        RunLog.insert_many(run_logs_to_update).execute()

        return local_db_prompt_model_list
    else:
        pass


def update_chat_model_changelog(
    action: ChangeLogAction,
    project_status: dict,
    uuid_list: List[str],
    local_db_chat_model_list: List[Dict],
    local_code_chat_model_name_list: List[str],
):
    if action == ChangeLogAction.ADD.value:
        chat_model_list = [
            x for x in project_status["chat_models"] if x["uuid"] in uuid_list
        ]
        for chat_model in chat_model_list:
            local_db_chat_model_name_list = [
                x["name"] for x in local_db_chat_model_list
            ]

            if chat_model["name"] not in local_db_chat_model_name_list:
                # IF chat_model not in Local DB
                if chat_model["name"] in local_code_chat_model_name_list:
                    # IF chat_model in Local Code
                    chat_model["used_in_code"] = True
                    chat_model["is_deployed"] = True
                else:
                    chat_model["used_in_code"] = False
                    chat_model["is_deployed"] = True
                ChatModel.create(**chat_model)
            else:
                local_db_chat_model = [
                    x
                    for x in local_db_chat_model_list
                    if x["name"] == chat_model["name"]
                ][0]
                if local_db_chat_model["is_deployed"] is False:
                    print(
                        "Creation of chatmodel with identical name was detected in local & deployment."
                    )
                    check_same = inquirer.confirm(
                        message="Are they same chatmodel? [y/n]", default=False
                    ).execute()
                    # check_same = input()
                    # if check_same == "y":
                    #     check_same = True
                    # else:
                    #     check_same = False

                    if not check_same:
                        # rename & change name in Local DB
                        print(
                            f"Please rename local chatmodel {chat_model['name']} to continue"
                        )
                        validate_new_chatmodel_name = (
                            lambda name: name not in local_code_chat_model_name_list
                            and name not in local_db_chat_model_name_list
                        )
                        new_model_name = inquirer.text(
                            message="Enter the new chatmodel name:",
                            validate=lambda x: validate_new_chatmodel_name(x),
                            invalid_message="chatmodel name already exists.",
                        ).execute()
                        # new_model_name = input()
                        rename_chat_model(local_db_chat_model["uuid"], new_model_name)

                        print("We changed the name of chatmodel in local DB.")
                        print(
                            f"Please change the name of chatmodel '{chat_model['name']}' in your project code and restart."
                        )
                        # dev 꺼버리고, 수정하고 다시 키라고 명령.
                        return False, local_db_chat_model_list
                    update_chat_model_uuid(
                        local_db_chat_model["uuid"], chat_model["uuid"]
                    )
                    local_db_chat_model_list: list = [
                        model_to_dict(x, recurse=False) for x in [ChatModel.select()]
                    ]
    else:
        # TODO: add code DELETE, CHANGE, FIX later
        pass
    return True, local_db_chat_model_list


def update_chat_model_version_changelog(
    action: ChangeLogAction,
    project_status: dict,
    uuid_list: List[str],
    local_db_chat_model_list: List[Dict],
    local_code_chat_model_name_list: List[str],
) -> List[Dict[str, Any]]:
    if action == ChangeLogAction.ADD.value:
        # find chat_model_version in project_status['chat_model_versions'] where uuid in uuid_list
        chat_model_version_list_in_changelog = [
            x for x in project_status["chat_model_versions"] if x["uuid"] in uuid_list
        ]

        # check if chat_model_version['uuid'] is in local_db_chat_model_list
        local_db_version_uuid_list = [
            str(x.uuid) for x in list(ChatModelVersion.select())
        ]
        version_list_to_update = [
            x
            for x in chat_model_version_list_in_changelog
            if x["uuid"] not in local_db_version_uuid_list
        ]

        for chat_model_version in version_list_to_update:
            chat_model_version["status"] = ModelVersionStatus.CANDIDATE.value

        ChatModelVersion.insert_many(version_list_to_update).execute()

        return local_db_chat_model_list
    else:
        pass
