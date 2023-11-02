import os
import sys
import yaml
import asyncio
from typing import Any, Dict, Tuple, List, Union, Optional

from promptmodel.database.crud import (
    get_latest_version_prompts,
    get_deployed_prompts,
    update_deployed_cache,
)
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils import logger
from promptmodel.apis.base import APIClient, AsyncAPIClient


async def fetch_prompts(name) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """fetch prompts.

    Args:
        name (str): name of promtpmodel

    Returns:
        Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]: (prompts, version_detail)
    """
    # Check dev_branch activate
    config = read_config()
    if "dev_branch" in config and config["dev_branch"]["initializing"] == True:
        return [], {}
    elif "dev_branch" in config and config["dev_branch"]["online"] == True:
        # get prompt from local DB
        prompt_rows, version_detail = get_latest_version_prompts(name)
        if prompt_rows is None:
            return [], {}
        return [
            {"role": prompt.role, "content": prompt.content}
            for prompt in prompt_rows
        ], version_detail
    else:
        if (
            "project" in config
            and "use_cache" in config["project"]
            and config["project"]["use_cache"] == True
        ):
            # call update_local API in background task
            asyncio.create_task(update_deployed_db(config))
            # get prompt from local DB by ratio
            prompt_rows, version_detail = get_deployed_prompts(name)
            if prompt_rows is None:
                return [], {}
        else:
            await update_deployed_db(config) # wait for update local DB cache
            prompt_rows, version_detail = get_deployed_prompts(name)
            if prompt_rows is None:
                return [], {}

        return [
            {"role": prompt.role, "content": prompt.content} for prompt in prompt_rows
        ], version_detail


async def update_deployed_db(config):
    if "project" not in config or "version" not in config["project"]:
        cached_project_version = "0.0.0"
    else:
        cached_project_version = config["project"]["version"]
    try:
        res = await AsyncAPIClient.execute(
            method="GET",
            path="/check_update",
            params={"cached_version": cached_project_version},
            use_cli_key=False,
        )
        res = res.json()
        if res["need_update"]:
            # update local DB with res['project_status']
            project_status = res["project_status"]
            await update_deployed_cache(project_status)
            upsert_config({"version": res["version"]}, section="project")
        else:
            upsert_config({"version": res["version"]}, section="project")
    except Exception as exception:
        logger.error(f"Deployment cache update error: {exception}")


def set_inputs_to_prompts(inputs: Dict[str, Any], prompts: List[Dict[str, str]]):
    messages = [
        {"content": prompt["content"].format(**inputs), "role": prompt["role"]}
        for prompt in prompts
    ]
    return messages
