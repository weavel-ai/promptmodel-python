from typing import (
    List,
    Dict,
    Any,
    Optional,
)
from pydantic import BaseModel
from litellm import ModelResponse
from openai._models import BaseModel as OpenAIObject


class Usage(OpenAIObject):
    def __init__(
        self, prompt_tokens=None, completion_tokens=None, total_tokens=None, **params
    ):
        super(Usage, self).__init__(**params)
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if completion_tokens:
            self.completion_tokens = completion_tokens
        if total_tokens:
            self.total_tokens = total_tokens


class Message(OpenAIObject):
    def __init__(
        self,
        content="default",
        role="assistant",
        logprobs=None,
        function_call=None,
        **params
    ):
        super(Message, self).__init__(**params)
        self.content = content
        self.role = role
        self._logprobs = logprobs
        if function_call:
            self.function_call = FunctionCall(**function_call)


class Delta(OpenAIObject):
    def __init__(self, content=None, role=None, function_call=None, **params):
        super(Delta, self).__init__(**params)
        if content is not None:
            self.content = content
        if role:
            self.role = role
        if function_call:
            self.function_call = FunctionCall(**function_call)


class Choices(OpenAIObject):
    def __init__(self, finish_reason=None, index=0, message=None, **params):
        super(Choices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = finish_reason
        else:
            self.finish_reason = "stop"
        self.index = index
        if message is None:
            self.message = Message(content=None)
        else:
            self.message = message


class StreamingChoices(OpenAIObject):
    def __init__(
        self, finish_reason=None, index=0, delta: Optional[Delta] = None, **params
    ):
        super(StreamingChoices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = finish_reason
        else:
            self.finish_reason = None
        self.index = index
        if delta:
            self.delta = delta
        else:
            self.delta = Delta()


class FunctionCall(OpenAIObject):
    arguments: str
    name: str


class ChoiceDeltaFunctionCall(OpenAIObject):
    arguments: str
    name: str


class LLMResponse(OpenAIObject):
    api_response: Optional[ModelResponse] = None
    raw_output: Optional[str] = None
    parsed_outputs: Optional[Dict[str, str]] = None
    error: Optional[bool] = None
    error_log: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class LLMStreamResponse(OpenAIObject):
    api_response: Optional[ModelResponse] = None
    raw_output: Optional[str] = None
    parsed_outputs: Optional[Dict[str, str]] = None
    error: Optional[bool] = None
    error_log: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class FunctionSchema(BaseModel):
    """
    {
            "name": str,
            "description": Optional[str],
            "parameters": {
                "type": "object",
                "properties": {
                    "argument_name": {
                        "type": str,
                        "description": Optional[str],
                        "enum": Optional[List[str]]
                    },
                },
                "required": Optional[List[str]],
            },
        }
    """

    class _Parameters(BaseModel):
        class _Properties(BaseModel):
            type: str
            description: Optional[str] = ""
            enum: Optional[List[str]] = []

        type: str = "object"
        properties: Dict[str, _Properties] = {}
        required: Optional[List[str]] = []

    name: str
    description: Optional[str] = None
    parameters: _Parameters
