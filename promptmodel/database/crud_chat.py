import json
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4, UUID
from promptmodel.database.models import (
    ChatModel,
    ChatModelVersion,
    ChatLogSession,
    ChatLog,
)
from playhouse.shortcuts import model_to_dict
from promptmodel.utils.enums import ModelVersionStatus, ParsingType
from promptmodel.utils.random_utils import select_version
from promptmodel.utils import logger
from promptmodel.database.config import db


def create_session(
    session_uuid: str,
    chat_model_version_uuid: Optional[str] = None,
) -> Optional[UUID]:
    try:
        return ChatLogSession.create(
            uuid=UUID(session_uuid),
            version_uuid=UUID(chat_model_version_uuid),
            run_from_deployment=False,
        ).uuid
    except Exception as e:
        logger.error(e)
        return None


def create_chat_logs(
    session_uuid: str,
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        # make ChatLog rows
        chat_log_rows = [
            {
                "session_uuid": UUID(session_uuid),
                "role": message["role"],
                "content": message["content"],
                "tool_calls": message["tool_calls"],
            }
            for message in messages
        ]
        with db.atomic():
            ChatLog.insert_many(chat_log_rows).execute()
    except Exception as e:
        logger.error(e)
        raise e


def fetch_chat_log_with_uuid(session_uuid: str):
    try:
        try:
            chat_logs: List[ChatLog] = (
                ChatLog.select()
                .where(ChatLog.session_uuid == UUID(session_uuid))
                .order_by(ChatLog.created_at.asc())
                .get()
            )
        except:
            return []

        chat_log_to_return = [
            {
                "role": chat_log.role,
                "content": chat_log.content,
                "tool_calls": chat_log.tool_calls,
            }
            for chat_log in chat_logs
        ]
        return chat_log_to_return
    except Exception as e:
        logger.error(e)
        return None


def get_latest_version_chat_model(
    chat_model_name: str,
    session_uuid: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    try:
        if session_uuid:
            chat_logs: List[ChatLog] = (
                ChatLog.select()
                .where(ChatLog.session_uuid == UUID(session_uuid))
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
