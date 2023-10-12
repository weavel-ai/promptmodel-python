from typing import Optional
from uuid import uuid4, UUID
from fastllm.database.models import (
    LLMModule,
    LLMModuleVersion,
    Prompt,
    RunLog,
    SampleInputs,
    DeployedLLMModule,
    DeployedLLMModuleVersion,
    DeployedPrompt
)
from peewee import Model
from fastllm.utils.enums import LLMModuleVersionStatus
from fastllm.utils.random_utils import select_version
from fastllm.database.config import db

# Insert
def create_llm_module(name: str, project_uuid: str):
    """Create a new LLM module with the given name."""
    return LLMModule.create(
        uuid=uuid4(),
        name=name,
        project_uuid=project_uuid
    )

def create_llm_modules(llm_module_list: list):
    """Create LLM modules with List of Dict"""
    with db.atomic():
        LLMModule.insert_many(llm_module_list).execute()
    return 
        

def create_llm_module_version(
    llm_module_uuid: str, from_uuid: Optional[str], status: str
):
    """Create a new LLM module version with the given parameters."""
    return LLMModuleVersion.create(
        uuid=uuid4(),
        from_uuid=from_uuid,
        llm_module_uuid=llm_module_uuid,
        status=status
    )

def create_llm_module_versions(llm_module_version_list: list):
    """Create LLM module versions with List of Dict"""
    with db.atomic():
        LLMModuleVersion.insert_many(llm_module_version_list).execute()
    return 

def create_prompt(
    llm_module_version_uuid: str,
    type: str,
    step: int,
    content: str,
):
    """Create a new prompt with the given parameters."""
    return Prompt.create(
        version_uuid=llm_module_version_uuid,
        type=type,
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
    is_deployment: bool,
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
def update_llm_module_version(
    llm_module_version_uuid: str,
    status: str
):
    """Update the status of the given LLM module version."""
    return LLMModuleVersion.update(
        status=status
    ).where(
        LLMModuleVersion.uuid == llm_module_version_uuid
    ).execute()

# Select all
def list_llm_modules():
    """List all LLM modules."""
    response = list(LLMModule.select())
    return [x.__data__ for x in response]


def list_llm_module_versions(llm_module_uuid: str):
    """List all LLM module versions for the given LLM module."""
    response = list(
        LLMModuleVersion.select().where(
            LLMModuleVersion.llm_module_uuid == llm_module_uuid
        )
    )
    return [x.__data__ for x in response]
    

def list_prompts(llm_module_version_uuid: str):
    """List all prompts for the given LLM module version."""
    response =  list(
        Prompt.select().where(Prompt.version_uuid == llm_module_version_uuid).order_by(Prompt.step)
    )
    return [x.__data__ for x in response]


def list_run_logs(llm_module_version_uuid: str):
    """List all run logs for the given LLM module version."""
    response = list(
        RunLog.select().where(RunLog.version_uuid == llm_module_version_uuid)
    )
    return [x.__data__ for x in response]

# Select one
def get_llm_module_uuid(llm_module_name: str):
    """Get uuid of llm module by name"""
    try:
        response = LLMModule.get(LLMModule.name == llm_module_name)
        return response.__data__
    except:
        return None

def get_sample_input(sample_name: str):
    """Get sample input from local DB"""
    try:
        response = SampleInputs.get(SampleInputs.name == sample_name)
        return response.__data__
    except:
        return None

def get_latest_version_prompts(llm_module_name: str):
    try:
        with db.atomic():
            latest_run_log: RunLog = (RunLog
                            .select()
                            .join(LLMModuleVersion)
                            .where(LLMModuleVersion.llm_module_uuid == LLMModule.get(LLMModule.name == llm_module_name).uuid)
                            .order_by(RunLog.created_at.desc())
                            .get())
            
            prompts = (Prompt
                        .select()
                        .where(Prompt.version_uuid == latest_run_log.version_uuid)
                        .order_by(Prompt.step.asc()))
            
            version: LLMModuleVersion = (
                LLMModuleVersion
                .select(LLMModuleVersion.model)
                .where(LLMModuleVersion.uuid == latest_run_log.version_uuid)
                .get()
            )

            return [prompt for prompt in prompts], version.model
            
    except Exception as e:
        return None, None

def get_deployed_prompts(llm_module_name: str):
    try:
        with db.atomic():
            versions: list[DeployedLLMModuleVersion] = (
                DeployedLLMModuleVersion
                .select()
                .join(DeployedLLMModule)
                .where(DeployedLLMModuleVersion.llm_module_uuid == DeployedLLMModule.get(DeployedLLMModule.name == llm_module_name).uuid)
                .get()
            )
            prompts: list[DeployedPrompt] = (
                DeployedPrompt
                .select()
                .where(DeployedPrompt.version_uuid.in_([version.uuid for version in versions]))
                .order_by(DeployedPrompt.step.asc())
            )
        # select version by ratio
        selected_version = select_version([version.__data__ for version in versions])
        selected_prompts = [prompt for prompt in prompts if prompt.version_uuid == selected_version['uuid']]
        
        return selected_prompts, selected_version['model']
    except Exception:
        return None, None

# Update
def update_is_deployment_llm_module(llm_module_uuid: str, is_deployment: bool):
    """Update the name of the given LLM module."""
    return LLMModule.update(
        is_deployment=is_deployment
    ).where(
        LLMModule.uuid == llm_module_uuid
    ).execute()
    
def update_local_usage_llm_module(llm_module_uuid: str, local_usage: bool):
    """Update the name of the given LLM module."""
    return LLMModule.update(
        local_usage=local_usage
    ).where(
        LLMModule.uuid == llm_module_uuid
    ).execute()
    
def update_local_usage_llm_module_by_name(llm_module_name: str, local_usage: bool):
    """Update the name of the given LLM module."""
    return LLMModule.update(
        local_usage=local_usage
    ).where(
        LLMModule.name == llm_module_name
    ).execute()
    
def hide_llm_module_not_in_code(
    local_llm_module_list: list
):
    return LLMModule.update(
        local_usage=False
    ).where(
        LLMModule.name.not_in(local_llm_module_list)
    ).execute()
    
async def update_deployed_cache(
    project_status: dict
):
    """Update Deployed Prompts Cache"""
    # TODO: 효율적으로 수정
    # 현재는 delete all & insert all
    llm_modules = project_status['llm_modules']
    llm_module_versions = project_status['llm_module_versions']
    for version in llm_module_versions:
        if version['is_published'] is True:
            version['ratio'] = 1.0
    prompts = project_status['prompts']
    
    with db.atomic():
        DeployedLLMModule.delete().execute()
        DeployedLLMModule.insert_many(llm_modules).execute()
        DeployedLLMModuleVersion.insert_many(llm_module_versions).execute()
        DeployedPrompt.insert_many(prompts).execute()
    return


def update_samples(
    samples: list[dict]
):
    """Update samples"""
    with db.atomic():
        SampleInputs.delete().execute()
        SampleInputs.insert_many(samples).execute()
    return