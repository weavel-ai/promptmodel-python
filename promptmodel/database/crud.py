from typing import Dict, List, Optional, Tuple
from promptmodel.database.models import (
    DeployedFunctionModel,
    DeployedFunctionModelVersion,
    DeployedPrompt,
)
from playhouse.shortcuts import model_to_dict
from promptmodel.utils.random_utils import select_version_by_ratio
from promptmodel.utils import logger
from promptmodel.database.config import db
from promptmodel.database.crud_chat import *


# Insert

# Select all

# Select one


def get_deployed_prompts(function_model_name: str) -> Tuple[List[DeployedPrompt], str]:
    try:
        with db.atomic():
            versions: List[DeployedFunctionModelVersion] = list(
                DeployedFunctionModelVersion.select()
                .join(DeployedFunctionModel)
                .where(
                    DeployedFunctionModelVersion.function_model_uuid
                    == DeployedFunctionModel.get(
                        DeployedFunctionModel.name == function_model_name
                    ).uuid
                )
            )
            prompts: List[DeployedPrompt] = list(
                DeployedPrompt.select()
                .where(
                    DeployedPrompt.version_uuid.in_(
                        [version.uuid for version in versions]
                    )
                )
                .order_by(DeployedPrompt.step.asc())
            )
        # select version by ratio
        selected_version = select_version_by_ratio(
            [version.__data__ for version in versions]
        )
        selected_prompts = list(
            filter(
                lambda prompt: str(prompt.version_uuid.uuid)
                == str(selected_version["uuid"]),
                prompts,
            )
        )

        version_details = {
            "model": selected_version["model"],
            "uuid": selected_version["uuid"],
            "parsing_type": selected_version["parsing_type"],
            "output_keys": selected_version["output_keys"],
        }

        return selected_prompts, version_details
    except Exception as e:
        logger.error(e)
        return None, None


# Update


async def update_deployed_cache(project_status: dict):
    """Update Deployed Prompts Cache"""
    # TODO: 효율적으로 수정
    # 현재는 delete all & insert all
    function_models = project_status["function_models"]
    function_model_versions = project_status["function_model_versions"]
    for version in function_model_versions:
        if version["is_published"] is True:
            version["ratio"] = 1.0
    prompts = project_status["prompts"]

    with db.atomic():
        DeployedFunctionModel.delete().execute()
        DeployedFunctionModelVersion.delete().execute()
        DeployedPrompt.delete().execute()
        DeployedFunctionModel.insert_many(function_models).execute()
        DeployedFunctionModelVersion.insert_many(function_model_versions).execute()
        DeployedPrompt.insert_many(prompts).execute()
    return
