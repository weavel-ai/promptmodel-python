from peewee import SqliteDatabase, Model

db = SqliteDatabase("./.promptmodel/promptmodel.db")


class BaseModel(Model):
    class Meta:
        database = db
