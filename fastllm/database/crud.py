from typing import Optional
from uuid import uuid4
from .models import LLMModule, LLMModuleVersion, Prompt, RunLog


def create_llm_module(name: str):
    """Create a new LLM module with the given name."""
    return LLMModule.create(uuid=uuid4(), name=name)


def create_llm_module_version(
    llm_module: LLMModule, version: int, from_version: Optional[int], is_working: bool
):
    """Create a new LLM module version with the given parameters."""
    return LLMModuleVersion.create(
        uuid=uuid4(),
        version=version,
        from_version=from_version,
        is_working=is_working,
        llm_module_uuid=llm_module.uuid,
    )


def create_prompt(
    llm_module_version: LLMModuleVersion,
    type: str,
    step: int,
    content: str,
):
    """Create a new prompt with the given parameters."""
    return Prompt.create(
        version_uuid=llm_module_version.uuid,
        type=type,
        step=step,
        content=content,
    )


def create_run_log(
    llm_module_version: LLMModuleVersion,
    inputs: str,
    raw_output: str,
    parsed_outputs: str,
    is_deployment: bool,
):
    """Create a new run log with the given parameters."""
    return RunLog.create(
        version_uuid=llm_module_version.uuid,
        inputs=inputs,
        raw_output=raw_output,
        parsed_outputs=parsed_outputs,
        is_deployment=is_deployment,
    )


def list_llm_modules():
    """List all LLM modules."""
    return LLMModule.select()


def list_llm_module_versions(llm_module: LLMModule):
    """List all LLM module versions for the given LLM module."""
    return LLMModuleVersion.select().where(
        LLMModuleVersion.llm_module_uuid == llm_module.uuid
    )
    

def list_prompts(llm_module_version: LLMModuleVersion):
    """List all prompts for the given LLM module version."""
    return Prompt.select().where(Prompt.version_uuid == llm_module_version.uuid).order_by(Prompt.step)


def list_run_logs(llm_module_version: LLMModuleVersion):
    """List all run logs for the given LLM module version."""
    return RunLog.select().where(RunLog.version_uuid == llm_module_version.uuid)
