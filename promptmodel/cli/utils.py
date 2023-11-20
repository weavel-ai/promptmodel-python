from typing import Any, Dict
from InquirerPy import inquirer
from rich import print
from promptmodel.apis.base import APIClient


def get_org(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gets the current organization from the configuration.

    :return: A dictionary containing the current organization.
    """
    if "user" not in config:
        print("User not logged in. Please run [violet]prompt login[/violet] first.")
        exit()
    if "default_org" not in config["user"]:
        orgs = APIClient.execute(method="GET", path="/list_orgs").json()
        choices = [
            {
                "key": org["name"],
                "name": org["name"],
                "value": org,
            }
            for org in orgs
        ]
        org = inquirer.select(message="Select organization:", choices=choices).execute()
    else:
        org = config["user"]["default_org"]
    return org


def get_project(config: Dict[str, Any], org: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gets the current project from the configuration.

    :return: A dictionary containing the current project.
    """
    if "default_project" not in config["user"]:
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
        project = inquirer.select(message="Select project:", choices=choices).execute()
    else:
        project = config["user"]["default_project"]
    return project
