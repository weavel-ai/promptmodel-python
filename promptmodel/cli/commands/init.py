from importlib import resources
import typer
from rich import print
from promptmodel.constants import (
    PROMPTMODEL_DEV_FILENAME,
    PROMPTMODEL_DEV_STARTER_FILENAME,
)


def init(from_cli: bool = True):
    """Initialize a new promptmodel project."""
    import os

    if not os.path.exists(PROMPTMODEL_DEV_FILENAME):
        # Read the content from the source file
        content = resources.read_text("promptmodel", PROMPTMODEL_DEV_STARTER_FILENAME)

        # Write the content to the target file
        with open(PROMPTMODEL_DEV_FILENAME, "w") as target_file:
            target_file.write(content)
        print(
            "[violet][bold]promptmodel_dev.py[/bold][/violet] was successfully created!"
        )
        print(
            "Add promptmodels in your code, then run [violet][bold]prompt dev[/bold][/violet] to start engineering prompts."
        )
    elif from_cli:
        print(
            "[yellow]promptmodel_dev.py[/yellow] was already initialized in this directory."
        )
        print(
            "Run [violet][bold]prompt dev[/bold][/violet] to start engineering prompts."
        )


app = typer.Typer(invoke_without_command=True, callback=init)
