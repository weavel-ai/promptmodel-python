from promptmodel.apis.base import APIClient
import typer
from InquirerPy import inquirer

from promptmodel.utils.config_utils import upsert_config


def configure():
    """Saves user's default organization and project."""

    orgs = APIClient.execute(method="GET", path="/organizations").json()
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
        path="/projects",
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

    upsert_config({"org": org, "project": project}, section="connection")


app = typer.Typer(invoke_without_command=True, callback=configure)
