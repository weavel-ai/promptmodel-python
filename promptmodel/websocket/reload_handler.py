
import sys
import importlib
from threading import Timer
from rich import print
from watchdog.events import FileSystemEventHandler

from promptmodel.apis.base import APIClient
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel import Client
from promptmodel.database.crud import (
    list_llm_modules,
    update_local_usage_llm_module_by_name,
    create_llm_modules,
    create_llm_module_versions,
    create_prompts,
    create_run_logs,
    update_samples
)
from promptmodel.utils.enums import (
    LLMModuleVersionStatus,
    ChangeLogAction,
)
from promptmodel.websocket.websocket_client import DevWebsocketClient

class CodeReloadHandler(FileSystemEventHandler):
    def __init__(
        self,
        _client_filename: str,
        _instance_name: str,
        dev_websocket_client: DevWebsocketClient,
    ):
        self._client_filename: str = _client_filename
        self.client_instance_name: str = _instance_name
        self.dev_websocket_client: DevWebsocketClient = dev_websocket_client  # 저장
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
        print(f"[violet]:dev:[/violet]  Reloading {self._client_filename} module due to changes...")
        # Reload the client module
        module_name = modified_file_path.replace("./", "").replace("/", ".")[
            :-3
        ]  # assuming the file is in the PYTHONPATH
        if module_name in sys.modules:
            # print(module_name)
            module = sys.modules[module_name]
            importlib.reload(module)

        reloaded_module = importlib.reload(sys.modules[self._client_filename])
        print(f"[violet]:dev:[/violet]  {self._client_filename} module reloaded successfully.")

        new_client_instance: Client = getattr(
            reloaded_module, self.client_instance_name
        )
        # print(new_client_instance.llm_modules)
        new_llm_module_name_list = [
            llm_module.name for llm_module in new_client_instance.llm_modules
        ]
        old_llm_module_name_list = [
            llm_module.name
            for llm_module in self.dev_websocket_client._client.llm_modules
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
