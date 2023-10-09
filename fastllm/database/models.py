from uuid import uuid4
from .config import BaseModel
from peewee import (
    CharField,
    DateTimeField,
    IntegerField,
    ForeignKeyField,
    BooleanField,
    UUIDField,
    TextField,
    AutoField,
)
import datetime


class LLMModule(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    name = CharField()


class LLMModuleVersion(BaseModel):
    uuid = UUIDField(unique=True, default=uuid4)
    created_at = DateTimeField(default=datetime.datetime.now)
    version = IntegerField()
    from_version = IntegerField(null=True)
    llm_module_uuid = ForeignKeyField(
        LLMModule,
        field=LLMModule.uuid,
        backref="versions",
        on_delete="CASCADE",
    )
    is_working = BooleanField(default=False)


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
