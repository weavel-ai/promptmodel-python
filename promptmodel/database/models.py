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

from promptmodel.database.config import BaseModel
from promptmodel.utils.enums import LLMModuleVersionStatus, ParsingType

class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        return json.loads(value)

class LLMModule(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    project_uuid = UUIDField()
    name = CharField()
    local_usage = BooleanField(default=True)
    is_deployment = BooleanField(
        default=False
    )  # mark if the module is published to the cloud


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
    parsing_type = CharField(
        null=True,
        default=None,
        constraints=[
            Check(
                f"parsing_type IN ('{ParsingType.COLON.value}', '{ParsingType.SQUARE_BRACKET.value}', '{ParsingType.DOUBLE_SQUARE_BRACKET.value}')"
            )
        ]
    )
    output_keys = JSONField(null=True, default=None)
    functions = JSONField(default=[])


class Prompt(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    version_uuid = ForeignKeyField(
        LLMModuleVersion,
        field=LLMModuleVersion.uuid,
        backref="prompts",
        on_delete="CASCADE",
    )
    role = CharField()
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
    inputs = JSONField(null=True, default={})
    raw_output = TextField()
    parsed_outputs = JSONField(null=True, default={})
    is_deployment = BooleanField(default=False)
    function_call = JSONField(null=True, default={})


class SampleInputs(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    name = TextField(unique=True)
    contents = JSONField()

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
    parsing_type = CharField(
        null=True,
        default=None,
        constraints=[
            Check(
                f"parsing_type IN ('{ParsingType.COLON.value}', '{ParsingType.SQUARE_BRACKET.value}', '{ParsingType.DOUBLE_SQUARE_BRACKET.value}')"
            )
        ]
    )
    output_keys = JSONField(null=True, default=None)
    functions = JSONField(default=[])

    

class DeployedPrompt(BaseModel):
    id = AutoField()
    version_uuid = ForeignKeyField(
        DeployedLLMModuleVersion,
        field=DeployedLLMModuleVersion.uuid,
        backref="prompts",
        on_delete="CASCADE",
    )
    role = CharField()
    step = IntegerField()
    content = TextField()
