from peewee import SqliteDatabase, Model

db = SqliteDatabase("././.db")


class BaseModel(Model):
    class Meta:
        database = db 
