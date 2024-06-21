from typing import (
    Dict,
    Any,
    Optional,
)
from pydantic import BaseModel
from litellm.utils import (
    ModelResponse,
)
from openai.types.chat.chat_completion import *


class ChatLogRequest(BaseModel):
    uuid: Optional[str] = None
    message: Dict[str, Any]
    metadata: Optional[Dict] = None
    api_response: Optional[ModelResponse] = None

    def __post_init__(
        self,
    ):
        if self.api_response is not None and self.message is None:
            self.message = self.api_response.choices[0].message.model_dump()


class RunLogRequest(BaseModel):
    uuid: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict] = None
    api_response: Optional[ModelResponse] = None
