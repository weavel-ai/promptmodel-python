"""CLI for project management."""

from promptmodel.apis.base import APIClient
import typer
from rich import print

from promptmodel.utils.config_utils import read_config, upsert_config
from ..utils import get_org

app = typer.Typer(no_args_is_help=True, short_help="Manage Client projects.")


@app.command()
def list():
    """List all projects."""
    config = read_config()
    org = get_org(config)
    projects = APIClient.execute(
        method="GET",
        path="/list_projects",
        params={"organization_id": org["organization_id"]},
    ).json()
    print("\nProjects:")
    for project in projects:
        print(f"📌 {project['name']} ({project['version']})")
        if project["description"]:
            print(f"   {project['description']}\n")
