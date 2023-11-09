import os
import sys
import importlib
from typing import Any, Dict, List
from threading import Timer
from rich import print
from watchdog.events import FileSystemEventHandler

from promptmodel.apis.base import APIClient
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils import logger
from promptmodel import DevApp, DevClient
from promptmodel.database.crud import (
    list_prompt_models,
    update_used_in_code_prompt_model_by_name,
    create_prompt_models,
    create_prompt_model_versions,
    create_prompts,
    create_run_logs,
    update_samples,
    get_prompt_model_uuid,
    update_prompt_model_uuid,
)
from promptmodel.utils.enums import (
    PromptModelVersionStatus,
    ChangeLogAction,
)
from promptmodel.websocket.websocket_client import DevWebsocketClient


class CodeReloadHandler(FileSystemEventHandler):
    def __init__(
        self,
        _devapp_filename: str,
        _instance_name: str,
        dev_websocket_client: DevWebsocketClient,
    ):
        self._devapp_filename: str = _devapp_filename
        self.devapp_instance_name: str = _instance_name
        self.dev_websocket_client: DevWebsocketClient = (
            dev_websocket_client  # save dev_websocket_client instance
        )
        self.timer = None

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if event.src_path.endswith(".py"):
            if self.timer:
                self.timer.cancel()
            # reload modified file & main file
            self.timer = Timer(0.5, self.reload_code, args=(event.src_path,))
            self.timer.start()

    def reload_code(self, modified_file_path: str):
        print(
            f"[violet]promptmodel:dev:[/violet]  Reloading {self._devapp_filename} module due to changes..."
        )
        relative_modified_path = os.path.relpath(modified_file_path, os.getcwd())
        # Reload the devapp module
        module_name = relative_modified_path.replace("./", "").replace("/", ".")[
            :-3
        ]  # assuming the file is in the PYTHONPATH

        if module_name in sys.modules:
            module = sys.modules[module_name]
            importlib.reload(module)

        reloaded_module = importlib.reload(sys.modules[self._devapp_filename])
        print(
            f"[violet]promptmodel:dev:[/violet]  {self._devapp_filename} module reloaded successfully."
        )

        new_devapp_instance: DevApp = getattr(
            reloaded_module, self.devapp_instance_name
        )
        # print(new_devapp_instance.prompt_models)
        new_prompt_model_name_list = [
            prompt_model.name for prompt_model in new_devapp_instance.prompt_models
        ]
        old_prompt_model_name_list = [
            prompt_model.name
            for prompt_model in self.dev_websocket_client._devapp.prompt_models
        ]

        # 사라진 prompt_models 에 대해 local db prompt_model.used_in_code False Update
        removed_name_list = list(
            set(old_prompt_model_name_list) - set(new_prompt_model_name_list)
        )
        for removed_name in removed_name_list:
            update_used_in_code_prompt_model_by_name(removed_name, False)

        # 새로 생긴 prompt_model 에 대해 local db prompt_model.used_in_code True Update
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
        # IF used_in_code=False 인 name=name 이 있을 경우, used_in_code=True
        update_by_changelog_for_reload(
            changelogs=changelogs,
            project_status=project_status,
            local_code_prompt_model_name_list=new_prompt_model_name_list,
        )

        for prompt_model in new_devapp_instance.prompt_models:
            if prompt_model.name not in old_prompt_model_name_list:
                update_used_in_code_prompt_model_by_name(prompt_model.name, True)

        # create prompt_models in local DB
        db_prompt_model_list = list_prompt_models()
        db_prompt_model_name_list = [x["name"] for x in db_prompt_model_list]
        only_in_local_names = list(
            set(new_prompt_model_name_list) - set(db_prompt_model_name_list)
        )
        only_in_local_prompt_models = [
            {"name": x, "project_uuid": project["uuid"]} for x in only_in_local_names
        ]
        create_prompt_models(only_in_local_prompt_models)

        # update samples in local DB
        update_samples(new_devapp_instance.samples)
        self.dev_websocket_client.update_devapp_instance(new_devapp_instance)


def update_by_changelog_for_reload(
    changelogs: List[Dict],
    project_status: dict,
    local_code_prompt_model_name_list: List[str],
):
    """Update Local DB by changelog"""
    local_db_prompt_model_list: list = list_prompt_models()  # {"name", "uuid"}

    for changelog in changelogs:
        level: int = changelog["level"]
        logs = changelog["logs"]
        if level == 1:
            for log in logs:
                subject = log["subject"]
                action: str = log["action"]
                if subject == "prompt_model":
                    local_db_prompt_model_list = update_prompt_model_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_prompt_model_list=local_db_prompt_model_list,
                        local_code_prompt_model_name_list=local_code_prompt_model_name_list,
                    )
                elif subject == "prompt_model_version":
                    local_db_prompt_model_list = update_prompt_model_version_changelog(
                        action=action,
                        project_status=project_status,
                        uuid_list=log["identifiers"],
                        local_db_prompt_model_list=local_db_prompt_model_list,
                        local_code_prompt_model_name_list=local_code_prompt_model_name_list,
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
                    create_prompt_models([prompt_model])
                else:
                    prompt_model["used_in_code"] = False
                    prompt_model["is_deployed"] = True
                    create_prompt_models([prompt_model])
            else:
                # Fix UUID of prompt_model
                local_uuid = get_prompt_model_uuid(prompt_model["name"])["uuid"]
                update_prompt_model_uuid(local_uuid, prompt_model["uuid"])
                local_db_prompt_model_list: list = list_prompt_models()
    else:
        # TODO: add code DELETE, CHANGE, FIX later
        pass

    return local_db_prompt_model_list


def update_prompt_model_version_changelog(
    action: ChangeLogAction,
    project_status: dict,
    uuid_list: List[str],
    local_db_prompt_model_list: List[Dict],
    local_code_prompt_model_name_list: List[str],
) -> List[Dict[str, Any]]:
    if action == ChangeLogAction.ADD.value:
        # find prompt_model_version in project_status['prompt_model_versions'] where uuid in uuid_list
        prompt_model_version_list_to_update = [
            x for x in project_status["prompt_model_versions"] if x["uuid"] in uuid_list
        ]
        # check if prompt_model_version['name'] is in local_code_prompt_model_list

        # find prompts and run_logs to update
        prompts_to_update = [
            x for x in project_status["prompts"] if x["version_uuid"] in uuid_list
        ]
        run_logs_to_update = [
            x for x in project_status["run_logs"] if x["version_uuid"] in uuid_list
        ]

        for prompt_model_version in prompt_model_version_list_to_update:
            prompt_model_version["status"] = PromptModelVersionStatus.CANDIDATE.value

        create_prompt_model_versions(prompt_model_version_list_to_update)
        create_prompts(prompts_to_update)
        create_run_logs(run_logs_to_update)

        # local_db_prompt_model_list += [{"name" : x['name'], "uuid" : x['uuid']} for x in prompt_model_version_list_to_update]
        return local_db_prompt_model_list
    else:
        pass
