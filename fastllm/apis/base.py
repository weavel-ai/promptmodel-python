import os
from typing import Dict
from ..utils.config_utils import read_config
import requests
from ..constants import ENDPOINT_URL
from ..utils.crypto import decrypt_message
from rich import print


class APIClient:
    """
    A class to represent an API request client.

    ...

    Methods
    -------
    get_headers():
        Generates headers for the API request.
    execute(method="GET", params=None, data=None, json=None, **kwargs):
        Executes the API request.
    """

    @classmethod
    def _get_headers(cls, use_cli_key: bool = True) -> Dict:
        """
        Reads, decrypts the api_key, and returns headers for API request.

        Returns
        -------
        dict
            a dictionary containing the Authorization header
        """
        config = read_config()
        if use_cli_key:
            if "user" not in config:
                print(
                    "User not logged in. Please run [violet]fastllm login[/violet] first."
                )
                exit()

            encrypted_key = config["user"]["encrypted_api_key"]
            if encrypted_key is None:
                raise Exception("No API key found. Please run 'fastllm login' first.")
            decrypted_key = decrypt_message(encrypted_key)
        else:
            decrypted_key = os.environ.get("FASTLLM_API_KEY")
        headers = {"Authorization": f"Bearer {decrypted_key}"}
        return headers

    @classmethod
    def execute(
        cls,
        path: str,
        method="GET",
        params: Dict = None,
        data: Dict = None,
        json: Dict = None,
        ignore_auth_error: bool = False,
        use_cli_key: bool = True,
        **kwargs,
    ) -> requests.Response:
        """
        Executes the API request with the decrypted API key in the headers.

        Parameters
        ----------
        method : str, optional
            The HTTP method of the request (default is "GET")
        params : dict, optional
            The URL parameters to be sent with the request
        data : dict, optional
            The request body to be sent with the request
        json : dict, optional
            The JSON-encoded request body to be sent with the request
        ignore_auth_error: bool, optional
            Whether to ignore authentication errors (default is False)
        **kwargs : dict
            Additional arguments to pass to the requests.request function

        Returns
        -------
        requests.Response
            The response object returned by the requests library
        """
        url = f"{ENDPOINT_URL}{path}"
        headers = cls._get_headers(use_cli_key)
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                data=data,
                json=json,
                **kwargs,
            )
            if not response:
                print(f"[red]Error: {response}[/red]")
                exit()
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                if not ignore_auth_error:
                    print(
                        "[red]Authentication failed. Please run [violet][bold]fastllm login[/bold][/violet] first.[/red]"
                    )
                    exit()
            else:
                print(f"[red]Error: {response}[/red]")
                exit()
        except requests.exceptions.ConnectionError:
            print("[red]Could not connect to the Fastllm API.[/red]")
        except requests.exceptions.Timeout:
            print("[red]The request timed out.[/red]")
