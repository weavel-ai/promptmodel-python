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


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        return json.loads(value)


# class DeployedChatModel(BaseModel):
#     uuid = UUIDField(unique=True, default=uuid4)
#     name = CharField()


# class DeployedChatModelVersion(BaseModel):
#     uuid = UUIDField(unique=True, default=uuid4)
#     from_uuid = UUIDField(null=True)
#     chat_model_uuid = ForeignKeyField(
#         DeployedChatModel,
#         field=DeployedChatModel.uuid,
#         backref="versions",
#         on_delete="CASCADE",
#     )
#     model = CharField()
#     is_published = BooleanField(default=False)
#     is_ab_test = BooleanField(default=False)
#     ratio = FloatField(null=True)
#     system_prompt = JSONField(null=True, default={})
#     functions = JSONField(default=[])
