import os 

from .models import (
    LLMModule,
    LLMModuleVersion,
    Prompt,
    RunLog,
    SampleInputs,
    DeployedLLMModule,
    DeployedLLMModuleVersion,
    DeployedPrompt
)
from .config import db


def initialize_db():
    if not os.path.exists("./.fastllm"):
        os.mkdir("./.fastllm")
    db.connect()
    with db.atomic():
        if not LLMModule.table_exists():
            db.create_tables(
                [
                    LLMModule,
                    LLMModuleVersion,
                    Prompt,
                    RunLog,
                    SampleInputs,
                    DeployedLLMModule,
                    DeployedLLMModuleVersion,
                    DeployedPrompt
                    ]
                )
    db.close()
