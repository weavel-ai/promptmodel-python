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


def get_latest_version_chat_model(
    chat_model_name: str,
    chat_uuid: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    try:
        if chat_uuid:
            # find ChatModelVersion by chat_uuid
            # Get ChatLog by chat_uuid
            # Get ChatModelVersion by ChatLog.version_uuid
            # Get ChatModelVersion.system_prompt

            return
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
