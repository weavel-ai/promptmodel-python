import time
from ...apis.base import APIClient
from requests import request
import typer
import webbrowser
from rich import print
from InquirerPy import inquirer

from ...constants import ENDPOINT_URL, WEB_CLIENT_URL
from ...utils.config_utils import read_config, upsert_config
from ..utils import get_org, get_project
from ...utils.crypto import generate_api_key, encrypt_message


def dev():
    """Creates a new prompt development environment, and opens up FastLLM in the browser."""
    config = read_config()
    org = get_org(config)
    project = get_project(config=config, org=org)
    # TODO: Init local database
    if "dev_branch" not in config:
        validate_branch_name = lambda name: APIClient.execute(
            method="GET", path="/check_dev_branch_name", params={"name": name}
        ).json()
        branch_name = inquirer.text(
            message="Enter a development branch name:",
            validate=lambda x: " " not in x and validate_branch_name(x),
            invalid_message="Branch name already exists or contains spaces.",
        ).execute()
        upsert_config({"name": branch_name}, section="dev_branch")
    else:
        branch_name = config["dev_branch"]["name"]
        # version_res = APIClient.execute(method="GET", path="/get_project_version", params={"uuid": project["uuid"]})
        # TODO: Check cloud changes since local dev_branch was created

    dev_url = f"{WEB_CLIENT_URL}/org/{org['slug']}/project/{project['uuid']}/dev/{branch_name}"
    print("\nCreating local development branch...")
    APIClient.execute(
        method="POST",
        path="/create_dev_branch",
        params={"name": branch_name, "project_uuid": project["uuid"]},
    )
    # TODO: Pull llm_module and versions from cloud
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
    # TODO: Open websocket connection to backend server


app = typer.Typer(invoke_without_command=True, callback=dev)
