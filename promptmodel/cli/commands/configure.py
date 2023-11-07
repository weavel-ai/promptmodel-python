import time
from ...apis.base import APIClient
from requests import request
import typer
import webbrowser
from rich import print
from InquirerPy import inquirer

from ...constants import ENDPOINT_URL, WEB_CLIENT_URL
from ...utils.config_utils import read_config, upsert_config
from ...utils.crypto import generate_api_key, encrypt_message


def configure():
    """Saves user's default organization and project."""

    orgs = APIClient.execute(method="GET", path="/list_orgs").json()
    choices = [
        {
            "key": org["name"],
            "name": org["name"],
            "value": org,
        }
        for org in orgs
    ]
    org = inquirer.select(
        message="Select default organization:", choices=choices
    ).execute()

    projects = APIClient.execute(
        method="GET",
        path="/list_projects",
        params={"organization_id": org["organization_id"]},
    ).json()
    choices = [
        {
            "key": project["name"],
            "name": project["name"],
            "value": project,
        }
        for project in projects
    ]
    project = inquirer.select(
        message="Select default project:", choices=choices
    ).execute()

    upsert_config({"default_org": org, "default_project": project}, section="user")


app = typer.Typer(invoke_without_command=True, callback=configure)
