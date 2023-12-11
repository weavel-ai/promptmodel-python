import os
import sys
import importlib
import asyncio
from typing import Any, Dict, List
from threading import Timer
from rich import print
from watchdog.events import FileSystemEventHandler
from playhouse.shortcuts import model_to_dict

from promptmodel.apis.base import APIClient
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils.async_utils import run_async_in_sync_threadsafe
from promptmodel.utils import logger
from promptmodel import DevApp
from promptmodel.database.models import (
    DeployedFunctionModel,
    DeployedFunctionModelVersion,
    DeployedPrompt,
)

from promptmodel.websocket.websocket_client import DevWebsocketClient
from promptmodel.types.enums import ServerTask


class CodeReloadHandler(FileSystemEventHandler):
    def __init__(
        self,
        _devapp_filename: str,
        _instance_name: str,
        dev_websocket_client: DevWebsocketClient,
        main_loop: asyncio.AbstractEventLoop,
    ):
        self._devapp_filename: str = _devapp_filename
        self.devapp_instance_name: str = _instance_name
        self.dev_websocket_client: DevWebsocketClient = (
            dev_websocket_client  # save dev_websocket_client instance
        )
        self.timer = None
        self.main_loop = main_loop

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if event.src_path.endswith(".py"):
            upsert_config({"reloading": True}, "connection")
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

        config = read_config()
        org = config["connection"]["org"]
        project = config["connection"]["project"]

        # save samples, FunctionSchema, FunctionModel, ChatModel to cloud server by websocket ServerTask request
        new_function_model_name_list = (
            new_devapp_instance._get_function_model_name_list()
        )
        new_chat_model_name_list = new_devapp_instance._get_chat_model_name_list()
        new_samples = new_devapp_instance.samples
        new_function_schemas = new_devapp_instance._get_function_schema_list()

        res = run_async_in_sync_threadsafe(
            self.dev_websocket_client.request(
                ServerTask.SYNC_CODE,
                message={
                    "new_function_model": new_function_model_name_list,
                    "new_chat_model": new_chat_model_name_list,
                    "new_samples": new_samples,
                    "new_schemas": new_function_schemas,
                },
            ),
            main_loop=self.main_loop,
        )

        # update_samples(new_devapp_instance.samples)
        upsert_config({"reloading": False}, "connection")
        self.dev_websocket_client.update_devapp_instance(new_devapp_instance)
