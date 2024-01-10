from __future__ import annotations

from dataclasses import dataclass
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
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.async_utils import run_async_in_sync
from promptmodel.utils.config_utils import check_connection_status_decorator
from promptmodel.types.response import (
    LLMStreamResponse,
    LLMResponse,
    FunctionModelConfig,
    UnitConfig
)
from promptmodel.types.enums import InstanceType
from promptmodel.apis.base import AsyncAPIClient
from promptmodel import DevClient


@dataclass
class FunctionModelInterface:
    name: str
    default_model: str


class RegisteringMeta(type):
    def __call__(cls, *args, **kwargs):
        instance: "FunctionModel" = super().__call__(*args, **kwargs)
        # Find the global client instance in the current context
        client: Optional[DevClient] = cls.find_client_instance()
        if client is not None:
            client.register_function_model(instance.name)
        return instance

    @staticmethod
    def find_client_instance():
        import sys

        # Get the current frame (frame where FunctionModel is called)
        frame = sys._getframe(2)
        # Get global variables in the current frame
        global_vars = frame.f_globals
        # Find an instance of DevClient among global variables
        for var_name, var_val in global_vars.items():
            if isinstance(var_val, DevClient):
                return var_val
        return None


class FunctionModel(metaclass=RegisteringMeta):
    """

    Args:
        name (_type_): _description_
        version (Optional[ Union[str, int] ], optional): Choose which FunctionModel version to use. Defaults to "deploy". It can be "deploy", "latest", or version number.
        api_key (Optional[str], optional): API key for the LLM. Defaults to None. If None, use api_key in .env file.
        unit_config (Optional[UnitConfig], optional): If it is not None, every logs from this FunctionModel will be connected to the UnitLogger. Defaults to None.
    """

    def __init__(
        self,
        name,
        version: Optional[
            Union[str, int]
        ] = "deploy",  # "deploy" or "latest" or version number
        api_key: Optional[str] = None,
        unit_config: Optional[UnitConfig] = None,
    ):
        self.name = name
        self.api_key = api_key
        self.unit_config = unit_config
        self.llm_proxy = LLMProxy(name, version, unit_config)
        self.version = version
        self.recent_log_uuid = None

    @check_connection_status_decorator
    def get_config(self, *args, **kwargs) -> FunctionModelConfig:
        """Get config for the promptmodel.
        It will fetch the prompt and version you specified from the Cloud. (It will be saved in cache DB, so there is no extra latency for API call.)
        - If you made A/B testing in Web Dashboard, it will fetch the prompt randomly by the A/B testing ratio.
        If dev mode is initializing, it will return None

        Returns:
            FunctionModelConfig: config for the promptmodel. It contains prompts and version_detail.
        """
        # add name to the list of function_models

        prompt, version_detail = run_async_in_sync(
            LLMProxy.fetch_prompts(self.name, self.version)
        )
        return FunctionModelConfig(
            prompts=prompt,
            model=version_detail["model"],
            name=self.name,
            version_uuid=str(version_detail["uuid"]),
            version=version_detail["version"],
            parsing_type=version_detail["parsing_type"]
            if "parsing_type" in version_detail
            else None,
            output_keys=version_detail["output_keys"]
            if "output_keys" in version_detail
            else None,
        )

    @check_connection_status_decorator
    def run(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Run FunctionModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = self.llm_proxy.run(inputs, functions, tools, self.api_key)
        self.recent_log_uuid = res.pm_detail.log_uuid
        return res

    @check_connection_status_decorator
    async def arun(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Async run FunctionModel. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find raw output in response.raw_output or response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = await self.llm_proxy.arun(
            inputs, functions, tools, self.api_key
        )
        self.recent_log_uuid = res.pm_detail.log_uuid
        return res

    @check_connection_status_decorator
    def stream(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run FunctionModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        cache: Optional[LLMStreamResponse] = None
        for item in self.llm_proxy.stream(inputs, functions, tools, self.api_key):
            yield item
            cache = item

        if cache:
            self.recent_log_uuid = cache.pm_detail.log_uuid

    @check_connection_status_decorator
    async def astream(
        self,
        inputs: Optional[Dict[str, Any]] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Coroutine[AsyncGenerator[LLMStreamResponse, None]]:
        """Async Run FunctionModel with stream=True. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find raw output in response.raw_output or response.api_response['choices'][0]['delta']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """

        async def async_gen():
            cache: Optional[LLMStreamResponse] = None
            async for item in self.llm_proxy.astream(
                inputs, functions, tools, self.api_key
            ):
                yield item
                cache = item
            if cache:
                self.recent_log_uuid = cache.pm_detail.log_uuid

        return async_gen()

    @check_connection_status_decorator
    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Run FunctionModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = self.llm_proxy.run_and_parse(
            inputs, functions, tools, self.api_key
        )
        self.recent_log_uuid = res.pm_detail.log_uuid
        return res

    @check_connection_status_decorator
    async def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Async Run FunctionModel and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): input to the promptmodel. Defaults to {}.

        Returns:
            LLMResponse: response from the promptmodel. you can find parsed outputs in response.parsed_outputs. You can also find raw output in response.api_response['choices'][0]['message']['content'].

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log.
        """
        res: LLMResponse = await self.llm_proxy.arun_and_parse(
            inputs, functions, tools, self.api_key
        )
        self.recent_log_uuid = res.pm_detail.log_uuid
        return res

    @check_connection_status_decorator
    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Generator[LLMStreamResponse, None, None]:
        """Run FunctionModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            Generator[LLMStreamResponse, None, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """
        cache: Optional[LLMStreamResponse] = None
        for item in self.llm_proxy.stream_and_parse(
            inputs, functions, tools, self.api_key
        ):
            yield item
            cache = item

        if cache:
            self.recent_log_uuid = cache.pm_detail.log_uuid

    @check_connection_status_decorator
    async def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Coroutine[AsyncGenerator[LLMStreamResponse, None]]:
        """Async Run FunctionModel with stream=True and make parsed outputs. It does not raise error.

        Args:
            inputs (Dict[str, Any], optional): _description_. Defaults to {}.

        Yields:
            AsyncGenerator[LLMStreamResponse, None]: Generator of LLMStreamResponse. you can find parsed outputs in response.parsed_outputs. You can also find raw output in reaponse.raw_output.

        Error:
            It does not raise error. If error occurs, you can check error in response.error and error_log in response.error_log
        """

        async def async_gen():
            cache: Optional[LLMStreamResponse] = None
            async for item in self.llm_proxy.astream_and_parse(
                inputs, functions, tools, self.api_key
            ):
                yield item
                cache = item
            if cache:
                self.recent_log_uuid = cache.pm_detail.log_uuid

        return async_gen()

    @check_connection_status_decorator
    async def log_score(
        self,
        log_uuid: Optional[str] = None,
        score: Optional[Dict] = {},
        *args,
        **kwargs,
    ) -> str:
        """Save Scores for RunLog of FunctionModel to the Cloud.

        Args:
            log_uuid (Optional[str], optional): UUID of the RunLog you want to save score. If None, it will use recent_log_uuid.
            score (Optional[float], optional): Scores of RunLog. Each keys will be created as Evaluation Metric in Promptmodel Dashboard. Defaults to {}.

        Returns:
            str: _description_
        """
        try:
            if log_uuid is None and self.recent_log_uuid:
                log_uuid = self.recent_log_uuid
            if log_uuid is None:
                raise Exception(
                    "log_uuid is None. Please run FunctionModel.run or FunctionModel.log first."
                )

            res = await AsyncAPIClient.execute(
                method="POST",
                path="/run_log_score",
                params={"run_log_uuid": log_uuid},
                json=score,
                use_cli_key=False,
            )
            if res.status_code != 200:
                logger.error(f"Logging error: {res}")
        except Exception as exception:
            logger.error(f"Logging error: {exception}")

    @check_connection_status_decorator
    async def log(
        self,
        version_uuid: str,
        openai_api_response: Optional[ModelResponse] = None,
        inputs: Optional[Dict[str, Any]] = {},
        parsed_outputs: Optional[Dict[str, Any]] = {},
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> str:
        """Save RunLog of FunctionModel to the Cloud.

        Args:
            version_uuid (str): version of the FunctionModel. You can find this in the FunctionModelConfig.
            openai_api_response (Optional[ModelResponse], optional): OpenAI Response Class. Defaults to None.
            inputs (Optional[Dict[str, Any]], optional): inputs dictionary for the FunctionModel. Defaults to {}.
            parsed_outputs (Optional[Dict[str, Any]], optional): parsed outputs. Defaults to {}.
            metadata (Optional[Dict[str, Any]], optional): metadata. You can save Any metadata for RunLog. Defaults to {}.

        Returns:
            str: UUID of the RunLog.
        """
        try:
            log_uuid = str(uuid4())
            content = {
                "uuid": log_uuid,
                "api_response": openai_api_response.model_dump(),
                "inputs": inputs,
                "parsed_outputs": parsed_outputs,
                "metadata": metadata,
            }

            config = kwargs["config"]
            if (
                "mask_inputs" in config["project"]
                and config["project"]["mask_inputs"] is True
            ):
                if "inputs" in content:
                    content["inputs"] = {
                        key: "PRIVATE LOGGING"
                        for key, value in content["inputs"].items()
                    }
            res = await AsyncAPIClient.execute(
                method="POST",
                path="/run_log",
                params={"version_uuid": version_uuid},
                json=content,
                use_cli_key=False,
            )
            if res.status_code != 200:
                logger.error(f"Logging error: {res}")
                
            if self.unit_config:
                res = await AsyncAPIClient.execute(
                    method="POST",
                    path="/unit/connect",
                    json={
                        "unit_log_uuid": self.unit_config.log_uuid,
                        "run_log_uuid": log_uuid,
                    },
                    use_cli_key=False,
                )
                if res.status_code != 200:
                    logger.error(f"Logging error: {res}")
                
            self.recent_log_uuid = log_uuid
            return log_uuid
        except Exception as exception:
            logger.error(f"Logging error: {exception}")


class PromptModel(FunctionModel):
    """Deprecated"""

    def __init__(self, name, version: Optional[Union[str, int]] = "deploy"):
        super().__init__(name, version)
