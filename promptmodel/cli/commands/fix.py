import typer
import os
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.database.orm import initialize_db


def fix():
    """Fix Bugs that can be occured in the promptmodel."""
    config = read_config()
    if "connection" in config:
        connection = config["connection"]
        if "initializing" in connection and connection["initializing"] == True:
            upsert_config({"initializing": False}, "connection")
        if "online" in connection and connection["online"] == True:
            upsert_config({"online": False}, "connection")
        if "reloading" in connection and connection["reloading"] == True:
            upsert_config({"reloading": False}, "connection")
    # delete .promptmodel/promptmodel.db
    # if .promptmodel/promptmodel.db exist, delete it
    if os.path.exists(".promptmodel/promptmodel.db"):
        os.remove(".promptmodel/promptmodel.db")
    # initialize_db()

    return


app = typer.Typer(invoke_without_command=True, callback=fix)
