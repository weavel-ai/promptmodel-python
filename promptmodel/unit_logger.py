from typing import (
    Dict,
    Optional,
)

import promptmodel.utils.logger as logger
from promptmodel.utils.async_utils import run_async_in_sync
from promptmodel.utils.config_utils import check_connection_status_decorator
from promptmodel.types.response import (
    UnitConfig,
)
from promptmodel.apis.base import AsyncAPIClient

class UnitLogger:
    def __init__(
        self,
        name: str,
        version: int
    ):
        self.name: str = name
        self.version: int = version
        self.config: Optional[UnitConfig] = None
        
    @check_connection_status_decorator
    def get_config(self) -> Optional[UnitConfig]:
        """Get the config of the component
        You can get the config directly from the UnitLogger.confg attribute.
        """
        return self.config
    
    @check_connection_status_decorator
    async def log_start(self, *args, **kwargs) -> Optional["UnitLogger"]:
        """Create Component Log on the cloud.
        It returns the UnitLogger itself, so you can use it like this:
        >>> component = UnitLogger("intent_classification_unit", 1).log_start()
        >>> res = FunctionModel("intent_classifier", unit_config=component.config).run(...)
        >>> component.log_score({"accuracy": 0.9})
        """
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/unit/log",
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
            self.config = UnitConfig(**res.json())
            
        return self
        
    
    @check_connection_status_decorator
    async def log_score(self, scores: Dict[str, float], *args, **kwargs):
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/unit/score",
            json={
                "unit_log_uuid" : self.config.log_uuid,
                "scores": scores,
            },
            use_cli_key=False
        )
        if res.status_code != 200:
            logger.error(f"Failed to log score for component {self.name} v{self.version}")
            
        return
    