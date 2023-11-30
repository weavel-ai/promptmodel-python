import os

from .models import (
    DeployedPromptModel,
    DeployedPromptModelVersion,
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
            if not DeployedPromptModel.table_exists():
                db.create_tables(
                    [
                        # PromptModel,
                        # PromptModelVersion,
                        # Prompt,
                        # RunLog,
                        # SampleInputs,
                        DeployedPromptModel,
                        DeployedPromptModelVersion,
                        DeployedPrompt,
                        # ChatModel,
                        # ChatModelVersion,
                        # ChatLogSession,
                        # ChatLog,
                    ]
                )
        db.close()
