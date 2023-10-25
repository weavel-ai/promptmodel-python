from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Generator,
    Optional,
    AsyncGenerator,
    Callable,
    Union
)
from pydantic import BaseModel
from litellm import ModelResponse as LiteLLMModelResponse

class ModelResponse(LiteLLMModelResponse):
    def __init__(self, **params):
        super(ModelResponse, self).__init__(**params)
    
class LLMResponse:
    def __init__(
        self,
        response: ModelResponse,
        raw_output: Optional[str] = None,
        parsed_outputs: Optional[Dict[str, str]] = None,
        error: Optional[bool] = None,
        error_log: Optional[str] = None,
        function_call: Optional[Dict[str, Any]] = None
    ):
        self.response = response
        self.raw_output = raw_output
        self.parsed_outputs = parsed_outputs
        self.error = error
        self.error_log = error_log
        self.function_call = function_call
    
class LLMStreamResponse:
    def __init__(
        self,
        response: Optional[ModelResponse] = None,
        raw_output: Optional[str] = None,
        parsed_outputs: Optional[Dict[str, str]] = None,
        error: Optional[bool] = None,
        error_log: Optional[str] = None,
        function_call: Optional[Dict[str, Any]] = None
    ):
        self.response = response
        self.raw_output = raw_output
        self.parsed_outputs = parsed_outputs
        self.error = error
        self.error_log = error_log
        self.function_call = function_call