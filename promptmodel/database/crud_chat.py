from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from promptmodel.database.models import (
    ChatModel,
    ChatModelVersion,
    ChatLog,
)
from playhouse.shortcuts import model_to_dict
from promptmodel.types.enums import ModelVersionStatus
from promptmodel.utils import logger
from promptmodel.database.config import db


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


def update_candidate_chat_model_version(new_candidates: dict):
    """Update candidate ChatModelVersion's candidate_version_id(version)"""
    with db.atomic():
        for uuid, version in new_candidates.items():
            (
                ChatModelVersion.update(version=version, is_deployed=True)
                .where(ChatModelVersion.uuid == uuid)
                .execute()
            )
        # Find ChatModel
        chat_model_versions: List[ChatModelVersion] = list(
            ChatModelVersion.select().where(
                ChatModelVersion.uuid.in_(list(new_candidates.keys()))
            )
        )
        chat_model_uuids = [
            chat_model.chat_model_uuid.uuid for chat_model in chat_model_versions
        ]
        ChatModel.update(is_deployed=True).where(
            ChatModel.uuid.in_(chat_model_uuids)
        ).execute()


def rename_chat_model(chat_model_uuid: str, new_name: str):
    """Update the name of the given ChatModel."""
    return (
        ChatModel.update(name=new_name)
        .where(ChatModel.uuid == chat_model_uuid)
        .execute()
    )


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


def find_ancestor_chat_model_version(
    chat_model_version_uuid: str, versions: Optional[list] = None
):
    """Find ancestor ChatModel version"""

    # get all versions
    if versions is None:
        response = list(ChatModelVersion.select())
        versions = [model_to_dict(x, recurse=False) for x in response]

    # find target version
    target = list(
        filter(lambda version: version["uuid"] == chat_model_version_uuid, versions)
    )[0]

    target = _find_ancestor(target, versions)

    return target


def find_ancestor_chat_model_versions(target_chat_model_uuid: Optional[str] = None):
    """find ancestor versions for each versions in input"""
    # get all versions
    if target_chat_model_uuid is not None:
        response = list(
            ChatModelVersion.select().where(
                ChatModelVersion.chat_model_uuid == target_chat_model_uuid
            )
        )
    else:
        response = list(ChatModelVersion.select())
    versions = [model_to_dict(x, recurse=False) for x in response]

    targets = list(
        filter(
            lambda version: version["status"] == ModelVersionStatus.CANDIDATE.value
            and version["version"] is None,
            versions,
        )
    )

    targets = [
        find_ancestor_chat_model_version(target["uuid"], versions) for target in targets
    ]
    targets_with_real_ancestor = [target for target in targets]

    return targets_with_real_ancestor


def _find_ancestor(target: dict, versions: List[Dict]):
    ancestor = None
    temp = target
    if target["from_uuid"] is None:
        ancestor = None
    else:
        while temp["from_uuid"] is not None:
            new_temp = [
                version for version in versions if version["uuid"] == temp["from_uuid"]
            ][0]
            if (
                new_temp["version"] is not None
                or new_temp["status"] == ModelVersionStatus.CANDIDATE.value
            ):
                ancestor = new_temp
                break
            else:
                temp = new_temp
        target["from_uuid"] = ancestor["uuid"] if ancestor is not None else None

    return target
