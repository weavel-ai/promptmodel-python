import time
import asyncio
import typer
import importlib
import signal
from typing import Dict, Any, List
from playhouse.shortcuts import model_to_dict

import webbrowser
from rich import print
from InquirerPy import inquirer
from watchdog.observers import Observer

from promptmodel import DevApp
from promptmodel.apis.base import APIClient
from promptmodel.constants import ENDPOINT_URL, WEB_CLIENT_URL
from promptmodel.cli.commands.init import init as promptmodel_init
from promptmodel.cli.utils import get_org, get_project
from promptmodel.cli.signal_handler import dev_terminate_signal_handler
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.websocket import DevWebsocketClient, CodeReloadHandler
from promptmodel.database.orm import initialize_db


def connect():
    """Connect websocket and opens up DevApp in the browser."""
    upsert_config({"initializing": True}, "connection")
    signal.signal(signal.SIGINT, dev_terminate_signal_handler)
    promptmodel_init(from_cli=False)

    config = read_config()

    if "project" not in config["connection"]:
        org = get_org(config)
        project = get_project(config=config, org=org)

        # connect
        res = APIClient.execute(
            method="POST",
            path="/connect_cli_project",
            params={"project_uuid": project["uuid"]},
        )
        if res.status_code != 200:
            print(f"Error: {res.json()['detail']}")
            return

        upsert_config(
            {
                "project": project,
                "org": org,
            },
            section="connection",
        )

    else:
        org = config["connection"]["org"]
        project = config["connection"]["project"]

        res = APIClient.execute(
            method="POST",
            path="/connect_cli_project",
            params={"project_uuid": project["uuid"]},
        )
        if res.status_code != 200:
            print(f"Error: {res.json()['detail']}")
            return

    _devapp_filename, devapp_instance_name = "promptmodel_dev:app".split(":")

    devapp_module = importlib.import_module(_devapp_filename)
    devapp_instance: DevApp = getattr(devapp_module, devapp_instance_name)

    dev_url = f"{WEB_CLIENT_URL}/org/{org['slug']}/projects/{project['uuid']}"

    # Open websocket connection to backend server
    dev_websocket_client = DevWebsocketClient(_devapp=devapp_instance)

    import threading

    try:
        main_loop = asyncio.get_running_loop()
    except RuntimeError:
        main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_loop)

    reloader_thread = threading.Thread(
        target=start_code_reloader,
        args=(_devapp_filename, devapp_instance_name, dev_websocket_client, main_loop),
    )
    reloader_thread.daemon = True  # Set the thread as a daemon
    reloader_thread.start()

    print(
        f"\nOpening [violet]Promptmodel[/violet] prompt engineering environment with the following configuration:\n"
    )
    print(f"📌 Organization: [blue]{org['name']}[/blue]")
    print(f"📌 Project: [blue]{project['name']}[/blue]")
    print(
        f"\nIf browser doesn't open automatically, please visit [link={dev_url}]{dev_url}[/link]"
    )
    webbrowser.open(dev_url)

    upsert_config({"online": True, "initializing": False}, section="connection")

    # save samples, FunctionSchema, PromptModel, ChatModel to cloud server in dev_websocket_client.connect_to_gateway

    res = APIClient.execute(
        method="POST",
        path="/save_instances_in_code",
        params={"project_uuid": project["uuid"]},
        json={
            "prompt_models": devapp_instance._get_prompt_model_name_list(),
            "chat_models": devapp_instance._get_chat_model_name_list(),
            "function_schemas": devapp_instance._get_function_schema_list(),
            "samples": devapp_instance.samples,
        },
        use_cli_key=False,
    )
    if res.status_code != 200:
        print(f"Error: {res.json()['detail']}")
        return

    # Open Websocket
    asyncio.run(
        dev_websocket_client.connect_to_gateway(
            project_uuid=project["uuid"],
            connection_name=project["name"],
            cli_access_header=APIClient._get_headers(),
        )
    )


app = typer.Typer(invoke_without_command=True, callback=connect)


def start_code_reloader(
    _devapp_filename, devapp_instance_name, dev_websocket_client, main_loop
):
    event_handler = CodeReloadHandler(
        _devapp_filename, devapp_instance_name, dev_websocket_client, main_loop
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
