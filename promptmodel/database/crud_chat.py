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


def create_chat_models(chat_model_list: List):
    """Create ChatModels with List of Dict"""
    with db.atomic():
        ChatModel.insert_many(chat_model_list).execute()
    return


def create_chat_model_versions(chat_model_version_list: List):
    """Creat ChatModel versions with List of Dict"""
    with db.atomic():
        ChatModelVersion.insert_many(chat_model_version_list).execute()
    return


def list_chat_models() -> List[Dict]:
    """List all ChatModels."""
    response: List[ChatModel] = list(ChatModel.select())
    return [model_to_dict(x, recurse=False) for x in response]


def list_chat_model_versions(chat_model_uuid: str) -> List[Dict]:
    """List all ChatModel versions for the given ChatModel."""
    response: List[ChatModelVersion] = list(
        ChatModelVersion.select()
        .where(ChatModelVersion.chat_model_uuid == chat_model_uuid)
        .order_by(ChatModelVersion.created_at)
    )
    return [model_to_dict(x, recurse=False) for x in response]


def get_chat_model_uuid(chat_model_name: str) -> Dict:
    """Get uuid of ChatModel by name"""
    try:
        response = ChatModel.get(ChatModel.name == chat_model_name)
        return model_to_dict(response, recurse=False)
    except:
        return None


def hide_chat_model_not_in_code(local_chat_model_list: List):
    return (
        ChatModel.update(used_in_code=False)
        .where(ChatModel.name.not_in(local_chat_model_list))
        .execute()
    )


def update_chat_model_uuid(local_uuid, new_uuid):
    """Update ChatModel.uuid"""
    if str(local_uuid) == str(new_uuid):
        return
    else:
        with db.atomic():
            local_chat_model: ChatModel = ChatModel.get(ChatModel.uuid == local_uuid)
            ChatModel.create(
                uuid=new_uuid,
                name=local_chat_model.name,
                project_uuid=local_chat_model.project_uuid,
                created_at=local_chat_model.created_at,
                used_in_code=local_chat_model.used_in_code,
                is_deployed=True,
            )
            ChatModelVersion.update(prompt_model_uuid=new_uuid).where(
                ChatModelVersion.chat_model_uuid == local_uuid
            ).execute()
            ChatModel.delete().where(ChatModel.uuid == local_uuid).execute()
        return


def update_used_in_code_chat_model_by_name(chatmodel_name: str, used_in_code: bool):
    """Update the name of the given ChatModel."""
    return (
        ChatModel.update(used_in_code=used_in_code)
        .where(ChatModel.name == chatmodel_name)
        .execute()
    )


def rename_chat_model(chat_model_uuid: str, new_name: str):
    """Update the name of the given ChatModel."""
    return (
        ChatModel.update(name=new_name)
        .where(ChatModel.uuid == chat_model_uuid)
        .execute()
    )


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
