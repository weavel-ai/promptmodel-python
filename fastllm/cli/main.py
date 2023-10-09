import typer
from .commands.login import app as login
from .commands.dev import app as dev
from .commands.project import app as project
from .commands.configure import app as configure


app = typer.Typer(no_args_is_help=True)

app.add_typer(login, name="login")
app.add_typer(dev, name="dev")
app.add_typer(project, name="project")
app.add_typer(configure, name="configure")


if __name__ == "__main__":
    app()
