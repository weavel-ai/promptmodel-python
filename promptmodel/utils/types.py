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
     
     
class PromptModelResponse:
    def __init__(self) -> None:
		   pass
