from uuid import uuid4
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Coroutine,
    Union,
)
from litellm import ModelResponse

import promptmodel.utils.logger as logger
from promptmodel.utils.async_utils import run_async_in_sync
from promptmodel.utils.config_utils import check_connection_status_decorator
from promptmodel.types.response import (
    LLMStreamResponse,
    LLMResponse,
    FunctionModelConfig,
    PromptComponentConfig,
)
from promptmodel.apis.base import AsyncAPIClient
from promptmodel import DevClient

class PromptComponent:
    def __init__(
        self,
        name: str,
        version: int
    ):
        self.name: str = name
        self.version: int = version
        self.config: Optional[PromptComponentConfig] = None
        
    @check_connection_status_decorator
    def get_config(self) -> Optional[PromptComponentConfig]:
        """Get the config of the component
        You can get the config directly from the PromptComponent.confg attribute.
        """
        return self.config
    
    @check_connection_status_decorator
    async def log_start(self, *args, **kwargs) -> Optional["PromptComponent"]:
        """Create Component Log on the cloud.
        It returns the PromptComponent itself, so you can use it like this:
        >>> component = PromptComponent("intent_classification_unit", 1).log_start()
        >>> res = FunctionModel("intent_classifier", prompt_component_config=component.config).run(...)
        """
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/prompt_component/log",
            json={
                "name": self.name,
                "version": self.version,
            },
            use_cli_key=False
        )
        if res.status_code != 200:
            logger.error(f"Failed to log start for component {self.name} v{self.version}")
            return None
        else:
            self.config = PromptComponentConfig(**res.json())
            
        return self
        
    
    @check_connection_status_decorator
    async def log_score(self, scores: Dict[str, float], *args, **kwargs):
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/prompt_component/score",
            json={
                "component_log_uuid" : self.config.log_uuid,
                "scores": scores,
            },
            use_cli_key=False
        )
        if res.status_code != 200:
            logger.error(f"Failed to log score for component {self.name} v{self.version}")
            
        return
    