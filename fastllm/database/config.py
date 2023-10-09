from peewee import SqliteDatabase, Model

db = SqliteDatabase("./.fastllm/fastllm.db")


class BaseModel(Model):
    class Meta:
        database = db 
