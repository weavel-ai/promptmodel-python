import os
from typing import Any, Dict
import yaml

CONFIG_FILE = "././config.yaml"


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
    if not os.path.exists("./."):
        os.mkdir("./.")

    with open(CONFIG_FILE, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False)