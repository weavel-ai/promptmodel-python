import json
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4, UUID
from promptmodel.database.models import (
    ChatModel,
    ChatModelVersion,
    ChatLog,
    DeployedChatModel,
    DeployedChatModelVersion,
)
from playhouse.shortcuts import model_to_dict
from promptmodel.utils.enums import ModelVersionStatus, ParsingType
from promptmodel.utils.random_utils import select_version
from promptmodel.utils import logger
from promptmodel.database.config import db


def create_chat_logs(
    chat_uuid: str,
    messages: List[Dict[str, Any]],
    chat_model_version_uuid: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        # get version_uuid from chatlog where chat_uuid = chat_uuid
        if chat_model_version_uuid is None:
            chat_logs: List[ChatLog] = (
                ChatLog.select()
                .where(ChatLog.uuid == UUID(chat_uuid))
                .order_by(ChatLog.created_at.desc())
                .get()
            )
            chat_model_version_uuid = chat_logs[0].version_uuid.uuid

        # make ChatLog rows
        chat_log_rows = [
            {
                "chat_uuid": UUID(chat_uuid),
                "version_uuid": UUID(chat_model_version_uuid),
                "role": message["role"],
                "content": message["content"],
                "function_call": message["function_call"],
                "run_from_deployed": False,
            }
            for message in messages
        ]
        with db.atomic():
            ChatLog.insert_many(chat_log_rows).execute()
    except Exception as e:
        logger.error(e)
        raise e


def fetch_chat_log_with_uuid(chat_uuid: str):
    try:
        try:
            chat_logs: List[ChatLog] = (
                ChatLog.select()
                .where(ChatLog.uuid == UUID(chat_uuid))
                .order_by(ChatLog.created_at.asc())
                .get()
            )
        except:
            return []

        chat_log_to_return = [
            {
                "role": chat_log.role,
                "content": chat_log.content,
                "function_call": chat_log.function_call,
            }
            for chat_log in chat_logs
        ]
        return chat_log_to_return
    except Exception as e:
        logger.error(e)
        return None


def get_latest_version_chat_model(
    chat_model_name: str,
    chat_uuid: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    try:
        if chat_uuid:
            chat_logs: List[ChatLog] = (
                ChatLog.select()
                .where(ChatLog.uuid == UUID(chat_uuid))
                .order_by(ChatLog.created_at.desc())
                .get()
            )
            chat_model_uuid = chat_logs[0].version_uuid.uuid

            version: ChatModelVersion = ChatModelVersion.get(
                ChatModelVersion.uuid == chat_model_uuid
            )

            instruction: List[Dict[str, Any]] = [version["system_prompt"]]
        else:
            with db.atomic():
                latest_chat_log: ChatLog = (
                    ChatLog.select()
                    .join(ChatModelVersion)
                    .where(
                        ChatModelVersion.chat_model_uuid
                        == ChatModel.get(ChatModel.name == chat_model_name).uuid
                    )
                    .order_by(ChatLog.created_at.desc())
                    .get()
                )

                version: ChatModelVersion = (
                    ChatModelVersion.select()
                    .where(ChatModelVersion.uuid == latest_chat_log.version_uuid.uuid)
                    .get()
                )

                instruction: List[Dict[str, Any]] = [version["system_prompt"]]

            version_details = {
                "model": version.model,
                "uuid": version.uuid,
            }

            return instruction, version_details

    except Exception as e:
        logger.error(e)
        return None, None
