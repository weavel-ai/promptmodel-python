import time
from ...apis.base import APIClient
from requests import request
import typer
import webbrowser
from rich import print

from ...constants import ENDPOINT_URL, GRANT_ACCESS_URL
from ...utils.config_utils import upsert_config
from ...utils.crypto import generate_api_key, encrypt_message


def login():
    """Authenticate FastLLM CLI."""
    # TODO: Check if already logged in
    api_key = generate_api_key()
    encrypted_key = encrypt_message(api_key)
    upsert_config({"encrypted_api_key": encrypted_key}, section="user")
    url = f"{GRANT_ACCESS_URL}?token={api_key}"
    webbrowser.open(url)
    print("Please grant access to the CLI by visiting the URL in your browser.")
    print("Once you have granted access, you can close the browser tab.")
    print(f"\nURL: [link={url}]{url}[/link]\n")
    print("Waiting...\n")
    waiting_time = 0
    while waiting_time < 300:
        # Check access every 5 seconds
        try:
            res = APIClient.execute("/check_cli_access", ignore_auth_error=True)
            if res.json() == True:
                print("[green]Access granted![/green] 🎉")
                print("Run [violet][bold]fastllm dev[/bold][/violet] to start developing prompts.\n")
                return
        except Exception as err:
            print(f"[red]Error: {err}[/red]")
        time.sleep(5)
        waiting_time += 5
    print("Please try again later.")


app = typer.Typer(invoke_without_command=True, callback=login)
