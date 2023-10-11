import os
import yaml
import asyncio
from typing import Any, Dict, Tuple, List

from fastllm.database.crud import (
    get_latest_version_prompts,
    get_deployed_prompts,
    update_deployed_cache
)
from fastllm.utils.config_utils import read_config, upsert_config
from fastllm.apis.base import APIClient

async def fetch_prompts(self, name) -> Tuple[List[Dict[str, str]], str]:
    # Check dev_branch activate
    config = read_config()
    if "dev_branch" in config and config["dev_branch"]['online'] == True:
        # get prompt from local DB
        prompt_rows, model = get_latest_version_prompts(name)
        return [{"role": prompt.role, "content" : prompt.content} for prompt in prompt_rows], model
    else:
        # call update_local API in background task
        asyncio.create_task(update_deployed_db(config))
        # get prompt from local DB by ratio
        prompt_rows, model = get_deployed_prompts(name)
        return [{"role": prompt.role, "content" : prompt.content} for prompt in prompt_rows], model
    

async def update_deployed_db(self, config):
    if "project" not in config or "version" not in config["project"]:
        cached_project_version = "0.0.0"
    else:
        cached_project_version = config["project"]["version"]
    
    res = APIClient.execute(
        method="GET",
        path="/check_update",
        params={"cached_version": cached_project_version},
        use_cli_key=False,
    ).json()
    
    if res['need_update']:
        # update local DB with res['project_status']
        project_status = res['project_status']
        await update_deployed_cache(project_status)
        upsert_config({"version": res['version']}, section="project")
    else:
        upsert_config({"version": res['version']}, section="project")

def set_inputs_to_prompts(
    inputs: Dict[str, Any],
    prompts: List[Dict[str, str]]
):
    messages = [{'content': prompt['content'].format(**inputs), 'role': prompt['role']} for prompt in prompts]
    return messages