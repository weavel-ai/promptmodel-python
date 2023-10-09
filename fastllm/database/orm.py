from .models import LLMModule, LLMModuleVersion, Prompt, RunLog
from .config import db


def initialize_db():
    db.connect()
    with db.atomic():
        if not LLMModule.table_exists():
            db.create_tables([LLMModule, LLMModuleVersion, Prompt, RunLog])
    db.close()
