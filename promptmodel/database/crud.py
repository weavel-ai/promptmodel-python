import json
from typing import Dict, List, Optional, Tuple
from uuid import uuid4, UUID
from promptmodel.database.models import (
    PromptModel,
    PromptModelVersion,
    Prompt,
    RunLog,
    SampleInputs,
    DeployedPromptModel,
    DeployedPromptModelVersion,
    DeployedPrompt,
)
from playhouse.shortcuts import model_to_dict
from promptmodel.types.enums import ModelVersionStatus, ParsingType
from promptmodel.utils.random_utils import select_version_by_ratio
from promptmodel.utils import logger
from promptmodel.database.config import db
from promptmodel.database.crud_chat import *


# Insert

# Select all

# Select one


def get_latest_version_prompts(prompt_model_name: str) -> Tuple[List[Prompt], str]:
    try:
        with db.atomic():
            latest_run_log: RunLog = (
                RunLog.select()
                .join(PromptModelVersion)
                .where(
                    PromptModelVersion.prompt_model_uuid
                    == PromptModel.get(PromptModel.name == prompt_model_name).uuid
                )
                .order_by(RunLog.created_at.desc())
                .get()
            )

            prompts: List[Prompt] = (
                Prompt.select()
                .where(Prompt.version_uuid == latest_run_log.version_uuid.uuid)
                .order_by(Prompt.step.asc())
            )

            version: PromptModelVersion = (
                PromptModelVersion.select()
                .where(PromptModelVersion.uuid == latest_run_log.version_uuid.uuid)
                .get()
            )

        version_details = {
            "model": version.model,
            "uuid": version.uuid,
            "parsing_type": version.parsing_type,
            "output_keys": version.output_keys,
        }

        return prompts, version_details

    except Exception as e:
        logger.error(e)
        return None, None


def get_deployed_prompts(prompt_model_name: str) -> Tuple[List[DeployedPrompt], str]:
    try:
        with db.atomic():
            versions: List[DeployedPromptModelVersion] = list(
                DeployedPromptModelVersion.select()
                .join(DeployedPromptModel)
                .where(
                    DeployedPromptModelVersion.prompt_model_uuid
                    == DeployedPromptModel.get(
                        DeployedPromptModel.name == prompt_model_name
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


def rename_prompt_model(prompt_model_uuid: str, new_name: str):
    """Update the name of the given PromptModel."""
    return (
        PromptModel.update(name=new_name)
        .where(PromptModel.uuid == prompt_model_uuid)
        .execute()
    )


def hide_prompt_model_not_in_code(local_prompt_model_list: list):
    return (
        PromptModel.update(used_in_code=False)
        .where(PromptModel.name.not_in(local_prompt_model_list))
        .execute()
    )


async def update_deployed_cache(project_status: dict):
    """Update Deployed Prompts Cache"""
    # TODO: 효율적으로 수정
    # 현재는 delete all & insert all
    prompt_models = project_status["prompt_models"]
    prompt_model_versions = project_status["prompt_model_versions"]
    for version in prompt_model_versions:
        if version["is_published"] is True:
            version["ratio"] = 1.0
    prompts = project_status["prompts"]

    with db.atomic():
        DeployedPromptModel.delete().execute()
        DeployedPromptModelVersion.delete().execute()
        DeployedPrompt.delete().execute()
        DeployedPromptModel.insert_many(prompt_models).execute()
        DeployedPromptModelVersion.insert_many(prompt_model_versions).execute()
        DeployedPrompt.insert_many(prompts).execute()
    return


def update_samples(samples: List[Dict]):
    """Update samples"""

    with db.atomic():
        SampleInputs.delete().execute()
        SampleInputs.insert_many(samples).execute()
    return


def update_prompt_model_uuid(local_uuid, new_uuid):
    """Update prompt_model_uuid"""
    if str(local_uuid) == str(new_uuid):
        return
    else:
        with db.atomic():
            local_prompt_model: PromptModel = PromptModel.get(
                PromptModel.uuid == local_uuid
            )
            PromptModel.create(
                uuid=new_uuid,
                name=local_prompt_model.name,
                project_uuid=local_prompt_model.project_uuid,
                created_at=local_prompt_model.created_at,
                used_in_code=local_prompt_model.used_in_code,
                is_deployed=True,
            )
            PromptModelVersion.update(prompt_model_uuid=new_uuid).where(
                PromptModelVersion.prompt_model_uuid == local_uuid
            ).execute()
            PromptModel.delete().where(PromptModel.uuid == local_uuid).execute()
        return


def find_ancestor_version(
    prompt_model_version_uuid: str, versions: Optional[list] = None
):
    """Find ancestor version"""

    # get all versions
    if versions is None:
        response = list(PromptModelVersion.select())
        versions = [model_to_dict(x, recurse=False) for x in response]

    # find target version
    target = list(
        filter(lambda version: version["uuid"] == prompt_model_version_uuid, versions)
    )[0]

    target = _find_ancestor(target, versions)

    prompts = list(Prompt.select().where(Prompt.version_uuid == target["uuid"]))
    prompts = [model_to_dict(x, recurse=False) for x in prompts]
    return target, prompts


def find_ancestor_versions(target_prompt_model_uuid: Optional[str] = None):
    """find ancestor versions for each versions in input"""
    # get all versions
    if target_prompt_model_uuid is not None:
        response = list(
            PromptModelVersion.select().where(
                PromptModelVersion.prompt_model_uuid == target_prompt_model_uuid
            )
        )
    else:
        response = list(PromptModelVersion.select())
    versions = [model_to_dict(x, recurse=False) for x in response]

    targets = list(
        filter(
            lambda version: version["status"] == ModelVersionStatus.CANDIDATE.value
            and version["version"] is None,
            versions,
        )
    )

    target_and_prompts = [
        find_ancestor_version(target["uuid"], versions) for target in targets
    ]
    targets_with_real_ancestor = [
        target_and_prompt[0] for target_and_prompt in target_and_prompts
    ]
    target_prompts = []
    for target_and_prompt in target_and_prompts:
        target_prompts += target_and_prompt[1]

    return targets_with_real_ancestor, target_prompts


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


def update_candidate_prompt_model_version(new_candidates: dict):
    """Update candidate version"""
    with db.atomic():
        for uuid, version in new_candidates.items():
            (
                PromptModelVersion.update(version=version, is_deployed=True)
                .where(PromptModelVersion.uuid == uuid)
                .execute()
            )
        # Find PromptModel
        prompt_model_versions: List[PromptModelVersion] = list(
            PromptModelVersion.select().where(
                PromptModelVersion.uuid.in_(list(new_candidates.keys()))
            )
        )
        prompt_model_uuids = [
            prompt_model.prompt_model_uuid.uuid
            for prompt_model in prompt_model_versions
        ]
        PromptModel.update(is_deployed=True).where(
            PromptModel.uuid.in_(prompt_model_uuids)
        ).execute()
