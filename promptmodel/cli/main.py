import os
import sys
import typer
from promptmodel.cli.commands.login import app as login
from promptmodel.cli.commands.init import app as init
from promptmodel.cli.commands.dev import app as dev
from promptmodel.cli.commands.project import app as project
from promptmodel.cli.commands.configure import app as configure


# 현재 작업 디렉토리를 sys.path에 추가
current_working_directory = os.getcwd()
if current_working_directory not in sys.path:
    sys.path.append(current_working_directory)

app = typer.Typer(no_args_is_help=True)

app.add_typer(login, name="login")
app.add_typer(init, name="init")
app.add_typer(dev, name="dev")
app.add_typer(project, name="project")
app.add_typer(configure, name="configure")


if __name__ == "__main__":
    app()
