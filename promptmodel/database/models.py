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
from promptmodel.database.models_chat import *
from promptmodel.types.enums import ParsingType


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        return json.loads(value)


class DeployedPromptModel(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    name = CharField()


class DeployedPromptModelVersion(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    from_version = IntegerField(null=True)
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
