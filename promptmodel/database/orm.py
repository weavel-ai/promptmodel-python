import os

from .models import (
    DeployedFunctionModel,
    DeployedFunctionModelVersion,
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
            if not DeployedFunctionModel.table_exists():
                db.create_tables(
                    [
                        # FunctionModel,
                        # FunctionModelVersion,
                        # Prompt,
                        # RunLog,
                        # SampleInputs,
                        DeployedFunctionModel,
                        DeployedFunctionModelVersion,
                        DeployedPrompt,
                        # ChatModel,
                        # ChatModelVersion,
                        # ChatLogSession,
                        # ChatLog,
                    ]
                )
        db.close()
