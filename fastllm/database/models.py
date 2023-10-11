from enum import Enum
import datetime
import json

from uuid import uuid4
from peewee import (
    CharField,
    DateTimeField,
    IntegerField,
    ForeignKeyField,
    BooleanField,
    UUIDField,
    TextField,
    AutoField,
    FloatField,
    Check,
)

from fastllm.database.config import BaseModel
from fastllm.utils.enums import LLMModuleVersionStatus

class LLMModule(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    project_uuid = UUIDField()
    name = CharField()
    local_usage = BooleanField(default=True)
    is_deployment = BooleanField(default=False) # mark if the module is published to the cloud

class LLMModuleVersion(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    from_uuid = UUIDField(null=True)
    llm_module_uuid = ForeignKeyField(
        LLMModule,
        field=LLMModule.uuid,
        backref="versions",
        on_delete="CASCADE",
    )
    status = CharField(
        constraints=[
            Check(
                f"status IN ('{LLMModuleVersionStatus.BROKEN.value}', '{LLMModuleVersionStatus.WORKING.value}', '{LLMModuleVersionStatus.CANDIDATE.value}')"
            )
        ]
    )
    model = CharField()
    candidate_version = IntegerField(null=True)
    is_published = BooleanField(default=False)


class Prompt(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    version_uuid = ForeignKeyField(
        LLMModuleVersion,
        field=LLMModuleVersion.uuid,
        backref="prompts",
        on_delete="CASCADE",
    )
    type = CharField()
    step = IntegerField()
    content = TextField()


class RunLog(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    version_uuid = ForeignKeyField(
        LLMModuleVersion,
        field=LLMModuleVersion.uuid,
        backref="run_logs",
        on_delete="CASCADE",
    )
    inputs = TextField()
    raw_output = TextField()
    parsed_outputs = TextField()
    is_deployment = BooleanField(default=False)
    
class SampleInputs(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    name = TextField(unique=True)
    contents = TextField()
    
    def set_contents(self, data: dict):
        self.contents = json.dumps(data)

    def get_contents(self) -> dict:
        return json.loads(self.contents)


class DeployedLLMModule(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    name = CharField()
    
class DeployedLLMModuleVersion(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    from_uuid = UUIDField(null=True)
    llm_module_uuid = ForeignKeyField(
        DeployedLLMModule,
        field=DeployedLLMModule.uuid,
        backref="versions",
        on_delete="CASCADE",
    )
    model = CharField()
    is_published = BooleanField(default=False)
    is_ab_test = BooleanField(default=False)
    ratio = FloatField(null=True)
    

class DeployedPrompt(BaseModel):
    id = AutoField()
    version_uuid = ForeignKeyField(
        DeployedLLMModuleVersion,
        field=DeployedLLMModuleVersion.uuid,
        backref="prompts",
        on_delete="CASCADE",
    )
    type = CharField()
    step = IntegerField()
    content = TextField()
