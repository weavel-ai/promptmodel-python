import os

from .models import (
    LLMModule,
    LLMModuleVersion,
    Prompt,
    RunLog,
    SampleInputs,
    DeployedLLMModule,
    DeployedLLMModuleVersion,
    DeployedPrompt,
)
from .config import db


def initialize_db():
    if not os.path.exists("./.promptmodel"):
        os.mkdir("./.promptmodel")
    # Check if db connection exists
    if db.is_closed():
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
                        DeployedPrompt,
                    ]
                )
        db.close()
