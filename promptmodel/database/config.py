from peewee import SqliteDatabase, Model

db = SqliteDatabase("./.promptmodel/.db")

class BaseModel(Model):
    class Meta:
        database = db 
