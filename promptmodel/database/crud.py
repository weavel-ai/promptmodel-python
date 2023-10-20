import json
from typing import Dict, List, Optional, Tuple
from uuid import uuid4, UUID
from promptmodel.database.models import (
    LLMModule,
    LLMModuleVersion,
    Prompt,
    RunLog,
    SampleInputs,
    DeployedLLMModule,
    DeployedLLMModuleVersion,
    DeployedPrompt,
)
from peewee import Model, Case
from playhouse.shortcuts import model_to_dict
from promptmodel.utils.enums import LLMModuleVersionStatus
from promptmodel.utils.random_utils import select_version
from promptmodel.database.config import db
from rich import print


# Insert
def create_llm_module(name: str, project_uuid: str):
    """Create a new LLM module with the given name."""
    return LLMModule.create(uuid=uuid4(), name=name, project_uuid=project_uuid)


def create_llm_modules(llm_module_list: list):
    """Create LLM modules with List of Dict"""
    with db.atomic():
        LLMModule.insert_many(llm_module_list).execute()
    return


def create_llm_module_version(
    llm_module_uuid: str, from_uuid: Optional[str], status: str, model: str
):
    """Create a new LLM module version with the given parameters."""
    return LLMModuleVersion.create(
        uuid=uuid4(),
        from_uuid=from_uuid,
        llm_module_uuid=llm_module_uuid,
        status=status,
        model=model,
    )


def create_llm_module_versions(llm_module_version_list: list):
    """Create LLM module versions with List of Dict"""
    with db.atomic():
        LLMModuleVersion.insert_many(llm_module_version_list).execute()
    return


def create_prompt(
    version_uuid: str,
    role: str,
    step: int,
    content: str,
):
    """Create a new prompt with the given parameters."""
    return Prompt.create(
        version_uuid=version_uuid,
        role=role,
        step=step,
        content=content,
    )


def create_prompts(prompt_list: list):
    """Create prompts with List of Dict"""
    with db.atomic():
        Prompt.insert_many(prompt_list).execute()
    return


def create_run_log(
    llm_module_version_uuid: str,
    inputs: str,
    raw_output: str,
    parsed_outputs: str,
    is_deployment: bool = False,
):
    """Create a new run log with the given parameters."""
    return RunLog.create(
        version_uuid=llm_module_version_uuid,
        inputs=inputs,
        raw_output=raw_output,
        parsed_outputs=parsed_outputs,
        is_deployment=is_deployment,
    )


def create_run_logs(run_log_list: list):
    """Create run_logs with List of Dict"""
    with db.atomic():
        RunLog.insert_many(run_log_list).execute()
    return


# Update For delayed Insert
def update_llm_module_version(llm_module_version_uuid: str, status: str):
    """Update the status of the given LLM module version."""
    return (
        LLMModuleVersion.update(status=status)
        .where(LLMModuleVersion.uuid == llm_module_version_uuid)
        .execute()
    )


# Select all
def list_llm_modules() -> List[Dict]:
    """List all LLM modules."""
    response: List[LLMModule] = list(LLMModule.select())
    return [model_to_dict(x, recurse=False) for x in response]


def list_llm_module_versions(llm_module_uuid: str) -> List[Dict]:
    """List all LLM module versions for the given LLM module."""
    response: List[LLMModuleVersion] = list(
        LLMModuleVersion.select()
        .where(LLMModuleVersion.llm_module_uuid == llm_module_uuid)
        .order_by(LLMModuleVersion.created_at)
    )
    return [model_to_dict(x, recurse=False) for x in response]


def list_prompts(llm_module_version_uuid: str) -> List[Dict]:
    """List all prompts for the given LLM module version."""
    response: List[Prompt] = list(
        Prompt.select()
        .where(Prompt.version_uuid == llm_module_version_uuid)
        .order_by(Prompt.step)
    )
    return [model_to_dict(x, recurse=False) for x in response]


def list_samples():
    """List all samples"""
    response = list(SampleInputs.select())
    return [x.__data__ for x in response]


def list_run_logs(llm_module_version_uuid: str) -> List[RunLog]:
    """List all run logs for the given LLM module version."""
    response: List[RunLog] = list(
        RunLog.select()
        .where(RunLog.version_uuid == llm_module_version_uuid)
        .order_by(RunLog.created_at.desc())
    )
    return [model_to_dict(x, recurse=False) for x in response]


# Select one
def get_llm_module_uuid(llm_module_name: str) -> Dict:
    """Get uuid of llm module by name"""
    try:
        response = LLMModule.get(LLMModule.name == llm_module_name)
        return model_to_dict(response, recurse=False)
    except:
        return None


def get_sample_input(sample_name: str) -> Dict:
    """Get sample input from local DB"""
    try:
        response = SampleInputs.get(SampleInputs.name == sample_name)
        sample = model_to_dict(response, recurse=False)
        # sample["contents"] = json.loads(sample["contents"])
        return sample
    except:
        return None


def get_latest_version_prompts(llm_module_name: str) -> Tuple[List[Prompt], str]:
    try:
        with db.atomic():
            latest_run_log: RunLog = (
                RunLog.select()
                .join(LLMModuleVersion)
                .where(
                    LLMModuleVersion.llm_module_uuid
                    == LLMModule.get(LLMModule.name == llm_module_name).uuid
                )
                .order_by(RunLog.created_at.desc())
                .get()
            )

            prompts: List[Prompt] = (
                Prompt.select()
                .where(Prompt.version_uuid == latest_run_log.version_uuid)
                .order_by(Prompt.step.asc())
            )

            version: LLMModuleVersion = (
                LLMModuleVersion.select(LLMModuleVersion.model)
                .where(LLMModuleVersion.uuid == latest_run_log.version_uuid)
                .get()
            )

            return prompts, version.model, version.uuid

    except Exception as e:
        return None, None, None


def get_deployed_prompts(llm_module_name: str) -> Tuple[List[DeployedPrompt], str]:
    try:
        with db.atomic():
            versions: List[DeployedLLMModuleVersion] = (
                DeployedLLMModuleVersion.select()
                .join(DeployedLLMModule)
                .where(
                    DeployedLLMModuleVersion.llm_module_uuid
                    == DeployedLLMModule.get(
                        DeployedLLMModule.name == llm_module_name
                    ).uuid
                )
                .get()
            )
            prompts: List[DeployedPrompt] = (
                DeployedPrompt.select()
                .where(
                    DeployedPrompt.version_uuid.in_(
                        [version.uuid for version in versions]
                    )
                )
                .order_by(DeployedPrompt.step.asc())
            )
        # select version by ratio
        selected_version = select_version([version.__data__ for version in versions])
        selected_prompts = list(
            filter(
                lambda prompt: prompt.version_uuid == selected_version["uuid"], prompts
            )
        )

        return selected_prompts, selected_version["model"], selected_version["uuid"]
    except Exception:
        return None, None, None


# Update
def update_is_deployment_llm_module(llm_module_uuid: str, is_deployment: bool):
    """Update the name of the given LLM module."""
    return (
        LLMModule.update(is_deployment=is_deployment)
        .where(LLMModule.uuid == llm_module_uuid)
        .execute()
    )


def update_local_usage_llm_module(llm_module_uuid: str, local_usage: bool):
    """Update the name of the given LLM module."""
    return (
        LLMModule.update(local_usage=local_usage)
        .where(LLMModule.uuid == llm_module_uuid)
        .execute()
    )


def update_local_usage_llm_module_by_name(llm_module_name: str, local_usage: bool):
    """Update the name of the given LLM module."""
    return (
        LLMModule.update(local_usage=local_usage)
        .where(LLMModule.name == llm_module_name)
        .execute()
    )

def rename_llm_module(llm_module_uuid: str, new_name: str):
    """Update the name of the given LLM module."""
    return (
        LLMModule.update(name=new_name)
        .where(LLMModule.uuid == llm_module_uuid)
        .execute()
    )

def hide_llm_module_not_in_code(local_llm_module_list: list):
    return (
        LLMModule.update(local_usage=False)
        .where(LLMModule.name.not_in(local_llm_module_list))
        .execute()
    )


async def update_deployed_cache(project_status: dict):
    """Update Deployed Prompts Cache"""
    # TODO: 효율적으로 수정
    # 현재는 delete all & insert all
    llm_modules = project_status["llm_modules"]
    llm_module_versions = project_status["llm_module_versions"]
    for version in llm_module_versions:
        if version["is_published"] is True:
            version["ratio"] = 1.0
    prompts = project_status["prompts"]

    with db.atomic():
        DeployedLLMModule.delete().execute()
        DeployedLLMModule.insert_many(llm_modules).execute()
        DeployedLLMModuleVersion.insert_many(llm_module_versions).execute()
        DeployedPrompt.insert_many(prompts).execute()
    return


def update_samples(samples: list[dict]):
    """Update samples"""
    for sample in samples:
        sample["contents"] = json.dumps(sample["contents"])

    with db.atomic():
        SampleInputs.delete().execute()
        SampleInputs.insert_many(samples).execute()
    return

def update_llm_module_uuid(local_uuid, new_uuid):
    """Update llm_module_uuid"""
    with db.atomic():
        local_llm_module : LLMModule = LLMModule.get(LLMModule.uuid == local_uuid)
        LLMModule.create(
            uuid=new_uuid,
            name=local_llm_module.name,
            project_uuid=local_llm_module.project_uuid,
            created_at=local_llm_module.created_at,
            local_usage=local_llm_module.local_usage,
            is_deployment=True
        )
        LLMModuleVersion.update(llm_module_uuid=new_uuid).where(LLMModuleVersion.llm_module_uuid == local_uuid).execute()
        LLMModule.delete().where(LLMModule.uuid == local_uuid).execute()
    return


def find_ancestor_version(
    llm_module_version_uuid: str, versions: Optional[list] = None
):
    """Find ancestor version"""

    # get all versions
    if versions is None:
        response = list(LLMModuleVersion.select())
        versions = [model_to_dict(x, recurse=False) for x in response]

    # find target version
    target = list(
        filter(lambda version: version["uuid"] == llm_module_version_uuid, versions)
    )[0]

    target = _find_ancestor(target, versions)

    prompts = list(Prompt.select().where(Prompt.version_uuid == target["uuid"]))
    prompts = [model_to_dict(x, recurse=False) for x in prompts]
    return target, prompts


def find_ancestor_versions(target_llm_module_uuid: Optional[str] = None):
    """find ancestor versions for each versions in input"""
    # get all versions
    if target_llm_module_uuid is not None:
        response = list(
            LLMModuleVersion.select().where(
                LLMModuleVersion.llm_module_uuid == target_llm_module_uuid
            )
        )
    else:
        response = list(LLMModuleVersion.select())
    versions = [model_to_dict(x, recurse=False) for x in response]

    print(versions)

    targets = list(
        filter(
            lambda version: version["status"] == LLMModuleVersionStatus.CANDIDATE.value
            and version["candidate_version"] is None,
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


def _find_ancestor(target: dict, versions: list[dict]):
    ancestor = None
    temp = target
    if target["from_uuid"] is None:
        ancestor = None
    else:
        while temp["from_uuid"] is not None:
            new_temp = [
                version for version in versions if version["uuid"] == temp["from_uuid"]
            ][0]
            if new_temp["candidate_version"] is not None:
                ancestor = new_temp
                break
            else:
                temp = new_temp
        target["from_uuid"] = ancestor["uuid"] if ancestor is not None else None

    print(f"temp: {temp}")
    return target


def update_candidate_version(new_candidates: dict):
    """Update candidate version"""
    with db.atomic():
        for uuid, version in new_candidates.items():
            (
                LLMModuleVersion.update(candidate_version=version)
                .where(LLMModuleVersion.uuid == uuid)
                .execute()
            )
