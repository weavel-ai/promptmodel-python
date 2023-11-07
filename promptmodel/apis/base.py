import os
from typing import Dict

import requests

import httpx
from rich import print

from promptmodel.utils.config_utils import read_config
from promptmodel.constants import ENDPOINT_URL
from promptmodel.utils.crypto import decrypt_message


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
                    "User not logged in. Please run [violet]prompt login[/violet] first."
                )
                exit()

            encrypted_key = (
                config["user"]["encrypted_api_key"]
                if "encrypted_api_key" in config["user"]
                else None
            )
            if encrypted_key is None:
                raise Exception("No API key found. Please run 'prompt login' first.")
            decrypted_key = decrypt_message(encrypted_key)
        else:
            decrypted_key = os.environ.get("PROMPTMODEL_API_KEY")
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
                        "[red]Authentication failed. Please run [violet][bold]prompt login[/bold][/violet] first.[/red]"
                    )
                    exit()
            else:
                print(f"[red]Error: {response}[/red]")
                exit()
        except requests.exceptions.ConnectionError:
            print("[red]Could not connect to the Promptmodel API.[/red]")
        except requests.exceptions.Timeout:
            print("[red]The request timed out.[/red]")


class AsyncAPIClient:
    """
    A class to represent an Async API request client.
    Used in Deployment stage.

    ...

    Methods
    -------
    get_headers():
        Generates headers for the API request.
    execute(method="GET", params=None, data=None, json=None, **kwargs):
        Executes the API request.
    """

    @classmethod
    async def _get_headers(cls, use_cli_key: bool = True) -> Dict:
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
                    "User not logged in. Please run [violet]prompt login[/violet] first."
                )
                exit()

            encrypted_key = config["user"]["encrypted_api_key"]
            if encrypted_key is None:
                raise Exception("No API key found. Please run 'prompt login' first.")
            decrypted_key = decrypt_message(encrypted_key)
        else:
            decrypted_key = os.environ.get("PROMPTMODEL_API_KEY")
            if decrypted_key is None:
                raise Exception(
                    "PROMPTMODEL_API_KEY was not found in the current environment."
                )
        headers = {"Authorization": f"Bearer {decrypted_key}"}
        return headers

    @classmethod
    async def execute(
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
        headers = await cls._get_headers(use_cli_key)
        try:
            async with httpx.AsyncClient(http2=True) as _client:
                response = await _client.request(
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
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                if not ignore_auth_error:
                    print("[red]Authentication failed.[/red]")
            else:
                print(f"[red]Error: {response}[/red]")

            return response
        except requests.exceptions.ConnectionError:
            print("[red]Could not connect to the Promptmodel API.[/red]")
        except requests.exceptions.Timeout:
            print("[red]The request timed out.[/red]")
        except Exception as exception:
            print(f"[red]Error: {exception}[/red]")
