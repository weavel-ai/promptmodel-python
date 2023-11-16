import os
import sys
import yaml
import asyncio
from uuid import UUID
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Tuple, List, Union, Optional, Coroutine
from playhouse.shortcuts import model_to_dict
from litellm import token_counter

from promptmodel.apis.base import AsyncAPIClient
from promptmodel.database.models import ChatLog, ChatLogSession
from promptmodel.database.crud import (
    get_latest_version_chat_model,
)
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils import logger
from promptmodel.utils.random_utils import select_version
from promptmodel.promptmodel_init import CacheManager, update_deployed_db


async def fetch_chat_model(
    name: str, session_uuid: Optional[str] = None
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """fetch instruction and version detail

    Args:
        name (str): name of ChatModel

    Returns:
        Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]: (prompts, version_detail)
    """
    # Check dev_branch activate
    config = read_config()
    if "dev_branch" in config and config["dev_branch"]["initializing"] == True:
        return [], {}
    elif "dev_branch" in config and config["dev_branch"]["online"] == True:
        # get prompt from local DB
        instruction, version_detail = get_latest_version_chat_model(name, session_uuid)
        if version_detail is None:
            return [], {}
        return instruction, version_detail
    else:
        try:
            res_data = await AsyncAPIClient.execute(
                method="GET",
                path="/fetch_published_chat_model_version",
                params={"chat_model_name": name, "session_uuid": session_uuid},
                use_cli_key=False,
            )
            res_data = res_data.json()
        except Exception as e:
            raise e
        chat_model_versions = res_data["chat_model_versions"]

        for version in chat_model_versions:
            if version["is_published"] is True:
                version["ratio"] = 1.0
        selected_version = select_version(chat_model_versions)

        instruction = [selected_version["system_prompt"]]

        version_detail = {
            "model": selected_version["model"],
            "uuid": selected_version["uuid"],
        }

        return instruction, version_detail


async def fetch_chat_log(session_uuid: str) -> List[Dict[str, Any]]:
    """fetch conversation log for session_uuid and version detail

    Args:
        session_uuid (str): session_uuid

    Returns:
        List[Dict[str, Any]] : list of conversation log
    """
    config = read_config()
    if "dev_branch" in config and config["dev_branch"]["initializing"] == True:
        return []
    elif "dev_branch" in config and config["dev_branch"]["online"] == True:
        try:
            chat_log_rows: List[ChatLog] = (
                ChatLog.select()
                .where(ChatLog.session_uuid == UUID(session_uuid))
                .order_by(ChatLog.created_at.asc())
                .get()
            )
            chat_logs = [
                {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": message.tool_calls,
                }
                for message in chat_log_rows
            ]
        except:
            chat_logs = []
        return chat_logs
    else:
        try:
            res_data = await AsyncAPIClient.execute(
                method="GET",
                path="/fetch_chat_logs",
                params={"session_uuid": session_uuid},
                use_cli_key=False,
            )
            res_data = res_data.json()
        except Exception as e:
            raise e

        # filter out unnecessary data
        res_data = [
            {
                "role": message["role"],
                "content": message["content"],
                "function_call": message["function_call"],
            }
            for message in res_data["chat_logs"]
        ]
        return res_data
