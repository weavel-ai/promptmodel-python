import os
import asyncio
from typing import Any, Dict
import yaml
from functools import wraps

CONFIG_FILE = "./.promptmodel/config.yaml"


def read_config():
    """
    Reads the configuration from the given filename.

    :return: A dictionary containing the configuration.
    """
    if not os.path.exists(CONFIG_FILE):
        return {}

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file) or {}
    return config


def merge_dict(d1: Dict[str, Any], d2: Dict[str, Any]):
    """
    Merge two dictionaries recursively.

    :param d1: The first dictionary.
    :param d2: The second dictionary.
    :return: The merged dictionary.
    """
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            d1[key] = merge_dict(d1[key], value)
        else:
            d1[key] = value
    return d1


def upsert_config(new_config: Dict[str, Any], section: str = None):
    """
    Upserts the given configuration file with the given configuration.

    :param new_config: A dictionary containing the new configuration.
    :param section: The section of the configuration to update.
    """
    config = read_config()
    if section:
        config_section = config.get(section, {})
        new_config = {section: merge_dict(config_section, new_config)}
    config = merge_dict(config, new_config)
    # If . directory does not exist, create it
    if not os.path.exists("./.promptmodel"):
        os.mkdir("./.promptmodel")

    with open(CONFIG_FILE, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False)


def check_connection_status_decorator(method):
    if asyncio.iscoroutinefunction(method):

        @wraps(method)
        async def async_wrapper(self, *args, **kwargs):
            config = read_config()
            if "connection" in config and (
                (
                    "initializing" in config["connection"]
                    and config["connection"]["initializing"]
                )
                or (
                    "reloading" in config["connection"]
                    and config["connection"]["reloading"]
                )
            ):
                return
            else:
                if "config" not in kwargs:
                    kwargs["config"] = config
                return await method(self, *args, **kwargs)

        # async_wrapper.__name__ = method.__name__
        # async_wrapper.__doc__ = method.__doc__
        return async_wrapper
    else:

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            config = read_config()
            if "connection" in config and (
                (
                    "initializing" in config["connection"]
                    and config["connection"]["initializing"]
                )
                or (
                    "reloading" in config["connection"]
                    and config["connection"]["reloading"]
                )
            ):
                return
            else:
                return method(self, *args, **kwargs)

        # wrapper.__name__ = method.__name__
        # wrapper.__doc__ = method.__doc__
        return wrapper
