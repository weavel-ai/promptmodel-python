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
from promptmodel.utils.enums import PromptModelVersionStatus, ParsingType


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        return json.loads(value)


class PromptModel(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    project_uuid = UUIDField()
    name = CharField()
    used_in_code = BooleanField(default=True)
    is_deployed = BooleanField(
        default=False
    )  # mark if the module is pushed to the cloud


class PromptModelVersion(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    from_uuid = UUIDField(null=True)
    prompt_model_uuid = ForeignKeyField(
        PromptModel,
        field=PromptModel.uuid,
        backref="versions",
        on_delete="CASCADE",
    )
    status = CharField(
        constraints=[
            Check(
                f"status IN ('{PromptModelVersionStatus.BROKEN.value}', '{PromptModelVersionStatus.WORKING.value}', '{PromptModelVersionStatus.CANDIDATE.value}')"
            )
        ]
    )
    model = CharField()
    version = IntegerField(null=True)
    is_published = BooleanField(default=False)
    is_deployed = BooleanField(default=False)
    parsing_type = CharField(
        null=True,
        default=None,
        constraints=[
            Check(
                f"parsing_type IN ('{ParsingType.COLON.value}', '{ParsingType.SQUARE_BRACKET.value}', '{ParsingType.DOUBLE_SQUARE_BRACKET.value}')"
            )
        ],
    )
    output_keys = JSONField(null=True, default=None)
    functions = JSONField(default=[])


class Prompt(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    version_uuid = ForeignKeyField(
        PromptModelVersion,
        field=PromptModelVersion.uuid,
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
        PromptModelVersion,
        field=PromptModelVersion.uuid,
        backref="run_logs",
        on_delete="CASCADE",
    )
    inputs = JSONField(null=True, default={})
    raw_output = TextField()
    parsed_outputs = JSONField(null=True, default={})
    run_from_deployment = BooleanField(default=False)
    function_call = JSONField(null=True, default={})


class SampleInputs(BaseModel):
    id = AutoField()
    created_at = DateTimeField(default=datetime.datetime.now)
    name = TextField(unique=True)
    contents = JSONField()


class DeployedPromptModel(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    name = CharField()


class DeployedPromptModelVersion(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    from_uuid = UUIDField(null=True)
    prompt_model_uuid = ForeignKeyField(
        DeployedPromptModel,
        field=DeployedPromptModel.uuid,
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
        ],
    )
    output_keys = JSONField(null=True, default=None)
    functions = JSONField(default=[])


class DeployedPrompt(BaseModel):
    id = AutoField()
    version_uuid = ForeignKeyField(
        DeployedPromptModelVersion,
        field=DeployedPromptModelVersion.uuid,
        backref="prompts",
        on_delete="CASCADE",
    )
    role = CharField()
    step = IntegerField()
    content = TextField()
