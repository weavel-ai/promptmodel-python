from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

from uuid import UUID
from threading import Thread
from rich import print
from litellm.utils import ModelResponse, get_max_tokens
from promptmodel.llms.llm import LLM
from promptmodel.database.models import (
    DeployedPrompt,
    DeployedPromptModel,
    DeployedPromptModelVersion,
)
from promptmodel.database.crud import (
    get_deployed_prompts,
)
from promptmodel.promptmodel_init import CacheManager
from promptmodel.utils.config_utils import read_config, upsert_config
from promptmodel.utils.random_utils import select_version_by_ratio
from promptmodel.utils import logger
from promptmodel.utils.async_utils import run_async_in_sync
from promptmodel.utils.token_counting import (
    num_tokens_for_messages_for_each,
    num_tokens_from_functions_input,
)
from promptmodel.utils.output_utils import update_dict
from promptmodel.apis.base import AsyncAPIClient
from promptmodel.types.response import (
    LLMResponse,
    LLMStreamResponse,
    PromptModelConfig,
    ChatModelConfig,
)


class LLMProxy(LLM):
    def __init__(self, name: str, version: Optional[Union[str, int]] = "deploy"):
        super().__init__()
        self._name = name
        self.version = version

    def _wrap_gen(self, gen: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = run_async_in_sync(
                LLMProxy.fetch_prompts(self._name, self.version)
            )
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )
            # Call the generator with the arguments
            stream_response: Generator[LLMStreamResponse, None, None] = gen(**call_args)

            api_response = None
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            error_occurs = False
            error_log = None
            for item in stream_response:
                if (
                    item.api_response and "delta" not in item.api_response.choices[0]
                ):  # only get the last api_response, not delta response
                    api_response = item.api_response
                if item.parsed_outputs:
                    dict_cache = update_dict(dict_cache, item.parsed_outputs)
                if item.raw_output:
                    string_cache += item.raw_output
                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log

                if error_occurs:
                    # delete all promptmodel data in item
                    item.raw_output = None
                    item.parsed_outputs = None
                    item.function_call = None
                yield item

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            run_async_in_sync(
                self._async_log_to_cloud(
                    version_details["uuid"], inputs, api_response, dict_cache, metadata
                )
            )

        return wrapper

    def _wrap_async_gen(self, async_gen: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = await LLMProxy.fetch_prompts(
                self._name, self.version
            )
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )

            # Call async_gen with the arguments
            stream_response: AsyncGenerator[LLMStreamResponse, None] = async_gen(
                **call_args
            )

            api_response = None
            dict_cache = {}  # to store aggregated dictionary values
            string_cache = ""  # to store aggregated string values
            error_occurs = False
            error_log = None
            api_response: Optional[ModelResponse] = None
            async for item in stream_response:
                if (
                    item.api_response and "delta" not in item.api_response.choices[0]
                ):  # only get the last api_response, not delta response
                    api_response = item.api_response
                if item.parsed_outputs:
                    dict_cache = update_dict(dict_cache, item.parsed_outputs)
                if item.raw_output:
                    string_cache += item.raw_output
                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log
                yield item

            # # add string_cache in model_response
            # if api_response:
            #     if "message" not in api_response.choices[0]:
            #         api_response.choices[0].message = {}
            #     if "content" not in api_response.choices[0].message:
            #         api_response.choices[0].message["content"] = string_cache
            #         api_response.choices[0].message["role"] = "assistant"

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            await self._async_log_to_cloud(
                version_details["uuid"], inputs, api_response, dict_cache, metadata
            )

            # raise Exception("error_log")

        return wrapper

    def _wrap_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = run_async_in_sync(
                LLMProxy.fetch_prompts(self._name, self.version)
            )
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            if llm_response.parsed_outputs:
                run_async_in_sync(
                    self._async_log_to_cloud(
                        version_details["uuid"],
                        inputs,
                        llm_response.api_response,
                        llm_response.parsed_outputs,
                        metadata,
                    )
                )
            else:
                run_async_in_sync(
                    self._async_log_to_cloud(
                        version_details["uuid"],
                        inputs,
                        llm_response.api_response,
                        {},
                        metadata,
                    )
                )
            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None
            return llm_response

        return wrapper

    def _wrap_async_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(inputs: Dict[str, Any], **kwargs):
            prompts, version_details = await LLMProxy.fetch_prompts(
                self._name, self.version
            )  # messages, model, uuid = self._fetch_prompts()
            call_args = self._prepare_call_args(
                prompts, version_details, inputs, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = await method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }

            if llm_response.parsed_outputs:
                await self._async_log_to_cloud(
                    version_details["uuid"],
                    inputs,
                    llm_response.api_response,
                    llm_response.parsed_outputs,
                    metadata,
                )
            else:
                await self._async_log_to_cloud(
                    version_details["uuid"],
                    inputs,
                    llm_response.api_response,
                    {},
                    metadata,
                )

            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None
            return llm_response

        return async_wrapper

    def _wrap_chat(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(session_uuid: str, **kwargs):
            instruction, version_details, message_logs = run_async_in_sync(
                LLMProxy.fetch_chat_model(self._name, session_uuid, self.version)
            )

            call_args = self._prepare_call_args_for_chat(
                message_logs, version_details, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            if llm_response.api_response:
                metadata["api_response"] = llm_response.api_response.model_dump()
                metadata["token_usage"] = (
                    llm_response.api_response.usage
                    if isinstance(llm_response.api_response.usage, dict)
                    else llm_response.api_response.usage.model_dump()
                )
                metadata["latency"] = llm_response.api_response._response_ms

            run_async_in_sync(
                self._async_chat_log_to_cloud(
                    session_uuid,
                    [llm_response.api_response.choices[0].message.model_dump()],
                    version_details["uuid"],
                    [metadata],
                )
            )

            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None

            return llm_response

        return wrapper

    def _wrap_async_chat(self, method: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(session_uuid: str, **kwargs):
            (
                instruction,
                version_details,
                message_logs,
            ) = await LLMProxy.fetch_chat_model(self._name, session_uuid, self.version)

            call_args = self._prepare_call_args_for_chat(
                message_logs, version_details, kwargs
            )

            # Call the method with the arguments
            llm_response: LLMResponse = await method(**call_args)
            error_occurs = llm_response.error
            error_log = llm_response.error_log
            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            if llm_response.api_response:
                metadata["api_response"] = llm_response.api_response.model_dump()
                metadata["token_usage"] = (
                    llm_response.api_response.usage
                    if isinstance(llm_response.api_response.usage, dict)
                    else llm_response.api_response.usage.model_dump()
                )
                metadata["latency"] = llm_response.api_response._response_ms

            await self._async_chat_log_to_cloud(
                session_uuid,
                [llm_response.api_response.choices[0].message.model_dump()],
                version_details["uuid"],
                [metadata],
            )

            if error_occurs:
                # delete all promptmodel data in llm_response
                llm_response.raw_output = None
                llm_response.parsed_outputs = None
                llm_response.function_call = None

            return llm_response

        return async_wrapper

    def _wrap_chat_gen(self, gen: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(session_uuid: str, **kwargs):
            instruction, version_details, message_logs = run_async_in_sync(
                LLMProxy.fetch_chat_model(self._name, session_uuid, self.version)
            )

            call_args = self._prepare_call_args_for_chat(
                message_logs, version_details, kwargs
            )
            # Call the generator with the arguments
            stream_response: Generator[LLMStreamResponse, None, None] = gen(**call_args)

            api_response = None
            error_occurs = False
            error_log = None
            for item in stream_response:
                if (
                    item.api_response and "delta" not in item.api_response.choices[0]
                ):  # only get the last api_response, not delta response
                    api_response = item.api_response

                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log

                if error_occurs:
                    # delete all promptmodel data in item
                    item.raw_output = None
                    item.parsed_outputs = None
                    item.function_call = None

                yield item

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            run_async_in_sync(
                self._async_chat_log_to_cloud(
                    session_uuid,
                    [api_response.choices[0].message.model_dump()],
                    version_details["uuid"],
                    [metadata],
                )
            )

        return wrapper

    def _wrap_async_chat_gen(self, async_gen: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(session_uuid: str, **kwargs):
            (
                instruction,
                version_details,
                message_logs,
            ) = await LLMProxy.fetch_chat_model(self._name, session_uuid, self.version)

            call_args = self._prepare_call_args_for_chat(
                message_logs, version_details, kwargs
            )
            # Call the generator with the arguments
            stream_response: AsyncGenerator[LLMStreamResponse, None] = async_gen(
                **call_args
            )

            api_response = None
            error_occurs = False
            error_log = None
            async for item in stream_response:
                if (
                    item.api_response and "delta" not in item.api_response.choices[0]
                ):  # only get the last api_response, not delta response
                    api_response = item.api_response

                if item.error and not error_occurs:
                    error_occurs = True
                    error_log = item.error_log

                if error_occurs:
                    # delete all promptmodel data in item
                    item.raw_output = None
                    item.parsed_outputs = None
                    item.function_call = None

                yield item

            metadata = {
                "error_occurs": error_occurs,
                "error_log": error_log,
            }
            await self._async_chat_log_to_cloud(
                session_uuid,
                [api_response.choices[0].message.model_dump()],
                version_details["uuid"],
                [metadata],
            )

        return wrapper

    def _prepare_call_args(
        self,
        prompts: List[Dict[str, str]],
        version_detail: Dict[str, Any],
        inputs: Dict[str, Any],
        kwargs,
    ):
        stringified_inputs = {key: str(value) for key, value in inputs.items()}
        messages = [
            {
                "content": prompt["content"].format(**stringified_inputs),
                "role": prompt["role"],
            }
            for prompt in prompts
        ]
        call_args = {
            "messages": messages,
            "model": version_detail["model"] if version_detail else None,
            "parsing_type": version_detail["parsing_type"] if version_detail else None,
            "output_keys": version_detail["output_keys"] if version_detail else None,
        }
        if call_args["parsing_type"] is None:
            del call_args["parsing_type"]
            del call_args["output_keys"]

        if "functions" in kwargs:
            call_args["functions"] = kwargs["functions"]

        if "tools" in kwargs:
            call_args["tools"] = kwargs["tools"]

        if "api_key" in kwargs:
            call_args["api_key"] = kwargs["api_key"]
        return call_args

    def _prepare_call_args_for_chat(
        self,
        messages: List[Dict[str, Any]],
        version_detail: Dict[str, Any],
        kwargs,
    ):
        call_args = {}
        token_per_tools = 0
        if "functions" in kwargs:
            call_args["functions"] = kwargs["functions"]
            token_per_tools = num_tokens_from_functions_input(
                functions=kwargs["functions"],
                model=version_detail["model"] if version_detail else "gpt-3.5-turbo",
            )

        if "tools" in kwargs:
            call_args["tools"] = kwargs["tools"]
            token_per_tools = num_tokens_from_functions_input(
                functions=kwargs["tools"],
                model=version_detail["model"] if version_detail else "gpt-3.5-turbo",
            )

        # truncate messages to make length <= model's max length
        model_max_tokens = get_max_tokens(
            model=version_detail["model"] if version_detail else "gpt-3.5-turbo"
        )
        token_per_messages = num_tokens_for_messages_for_each(
            messages, version_detail["model"]
        )
        token_limit_exceeded = (
            sum(token_per_messages) + token_per_tools
        ) - model_max_tokens
        if token_limit_exceeded > 0:
            while token_limit_exceeded > 0:
                # erase the second oldest message (first one is system prompt, so it should not be erased)
                if len(messages) == 1:
                    # if there is only one message, Error cannot be solved. Just call LLM and get error response
                    break
                token_limit_exceeded -= token_per_messages[1]
                del messages[1]
                del token_per_messages[1]

        call_args["messages"] = messages
        call_args["model"] = version_detail["model"] if version_detail else None

        if "api_key" in kwargs:
            call_args["api_key"] = kwargs["api_key"]

        if "tools" in kwargs:
            call_args["tools"] = kwargs["tools"]

        return call_args

    async def _async_log_to_cloud(
        self,
        version_uuid: str,
        inputs: Optional[Dict] = None,
        api_response: Optional[ModelResponse] = None,
        parsed_outputs: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        # Perform the logging asynchronously
        if api_response:
            api_response_dict = api_response.model_dump()
            api_response_dict["response_ms"] = api_response._response_ms
        else:
            api_response_dict = None
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/log_deployment_run",
            params={
                "version_uuid": version_uuid,
            },
            json={
                "inputs": inputs,
                "api_response": api_response_dict,
                "parsed_outputs": parsed_outputs,
                "metadata": metadata,
            },
            use_cli_key=False,
        )
        if res.status_code != 200:
            print(f"[red]Failed to log to cloud: {res.json()}[/red]")
        return res

    async def _async_chat_log_to_cloud(
        self,
        session_uuid: str,
        messages: List[Dict[str, Any]],
        version_uuid: Optional[str] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        # Perform the logging asynchronously
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/log_deployment_chat",
            params={
                "session_uuid": session_uuid,
                "version_uuid": version_uuid,
            },
            json={
                "messages": messages,
                "metadata": metadata,
            },
            use_cli_key=False,
        )
        if res.status_code != 200:
            print(f"[red]Failed to log to cloud: {res.json()}[/red]")
        return res

    async def _async_make_session_cloud(
        self,
        session_uuid: str,
        version_uuid: Optional[str] = None,
    ):
        # Perform the logging asynchronously
        res = await AsyncAPIClient.execute(
            method="POST",
            path="/make_session",
            params={
                "session_uuid": session_uuid,
                "version_uuid": version_uuid,
            },
            use_cli_key=False,
        )
        if res.status_code != 200:
            print(f"[red]Failed to log to cloud: {res.json()}[/red]")
        return res

    def make_kwargs(self, **kwargs):
        res = {}
        for key, value in kwargs.items():
            if value is not None:
                res[key] = value
        return res

    def run(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_method(super().run)(inputs, **kwargs)

    def arun(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_async_method(super().arun)(inputs, **kwargs)

    def stream(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_gen(super().stream)(inputs, **kwargs)

    def astream(
        self,
        inputs: Optional[Dict[str, Any]] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_async_gen(super().astream)(inputs, **kwargs)

    def run_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_method(super().run_and_parse)(inputs, **kwargs)

    def arun_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_async_method(super().arun_and_parse)(inputs, **kwargs)

    def stream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> Generator[LLMStreamResponse, None, None]:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_gen(super().stream_and_parse)(inputs, **kwargs)

    def astream_and_parse(
        self,
        inputs: Dict[str, Any] = {},
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_async_gen(super().astream_and_parse)(inputs, **kwargs)

    def chat_run(
        self,
        session_uuid: str,
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_chat(super().run)(session_uuid, **kwargs)

    def chat_arun(
        self,
        session_uuid: str,
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_async_chat(super().arun)(session_uuid, **kwargs)

    def chat_stream(
        self,
        session_uuid: str,
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_chat_gen(super().stream)(session_uuid, **kwargs)

    def chat_astream(
        self,
        session_uuid: str,
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = self.make_kwargs(functions=functions, api_key=api_key, tools=tools)
        return self._wrap_async_chat_gen(super().astream)(session_uuid, **kwargs)

    @staticmethod
    async def fetch_prompts(
        name,
        version: Optional[Union[str, int]] = "deploy",
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """fetch prompts.

        Args:
            name (str): name of PromptModel

        Returns:
            Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]: (prompts, version_detail)
        """
        # Check connection activate
        config = read_config()
        if (
            "connection" in config
            and "initializing" in config["connection"]
            and config["connection"]["initializing"] == True
        ):
            return [], {}
        elif (
            "connection" in config
            and "reloading" in config["connection"]
            and config["connection"]["reloading"] == True
        ):
            return [], {}
        else:
            if (
                "project" in config
                and "use_cache" in config["project"]
                and config["project"]["use_cache"] == True
                and version == "deploy"
            ):
                cache_manager = CacheManager()
                # call update_local API in background task
                cache_update_thread = Thread(
                    target=cache_manager.cache_update_background_task, args=(config,)
                )
                cache_update_thread.daemon = True
                cache_update_thread.start()

                # get prompt from local DB by ratio
                prompt_rows, version_detail = get_deployed_prompts(name)
                if prompt_rows is None:
                    return [], {}

                return [
                    {"role": prompt.role, "content": prompt.content}
                    for prompt in prompt_rows
                ], version_detail

            else:
                try:
                    prompts_data = await AsyncAPIClient.execute(
                        method="GET",
                        path="/fetch_prompt_model_version",
                        params={"prompt_model_name": name, "version": version},
                        use_cli_key=False,
                    )
                    prompts_data = prompts_data.json()
                except Exception as e:
                    raise e
                prompt_model_versions = prompts_data["prompt_model_versions"]
                prompts = prompts_data["prompts"]
                if version == "deploy":
                    for version in prompt_model_versions:
                        if version["is_published"] is True:
                            version["ratio"] = 1.0
                    selected_version = select_version_by_ratio(prompt_model_versions)
                else:
                    selected_version = prompt_model_versions[0]

                prompt_rows = list(
                    filter(
                        lambda prompt: str(prompt["version_uuid"])
                        == str(selected_version["uuid"]),
                        prompts,
                    )
                )
                # sort prompt_rows by step
                prompt_rows = sorted(prompt_rows, key=lambda prompt: prompt["step"])

                version_detail = {
                    "model": selected_version["model"],
                    "uuid": selected_version["uuid"],
                    "parsing_type": selected_version["parsing_type"],
                    "output_keys": selected_version["output_keys"],
                }

                if prompt_rows is None:
                    return [], {}

                return [
                    {"role": prompt["role"], "content": prompt["content"]}
                    for prompt in prompt_rows
                ], version_detail

    @staticmethod
    async def fetch_chat_model(
        name: str,
        session_uuid: Optional[str] = None,
        version: Optional[Union[str, int]] = "deploy",
    ) -> Tuple[str, Dict[str, Any], List[Dict]]:
        """fetch instruction and version detail

        Args:
            name (str): name of ChatModel

        Returns:
            Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]: (prompts, version_detail)
        """
        # Check connection activate
        config = read_config()
        if (
            "connection" in config
            and "initializing" in config["connection"]
            and config["connection"]["initializing"] == True
        ):
            return "", {}, []
        elif (
            "connection" in config
            and "reloading" in config["connection"]
            and config["connection"]["reloading"] == True
        ):
            return "", {}, []
        else:
            try:
                res_data = await AsyncAPIClient.execute(
                    method="GET",
                    path="/fetch_chat_model_version_with_chat_log",
                    params={
                        "chat_model_name": name,
                        "session_uuid": session_uuid,
                        "version": version,
                    },
                    use_cli_key=False,
                )
                res_data = res_data.json()
            except Exception as e:
                raise e
            chat_model_versions = res_data["chat_model_versions"]

            if (
                session_uuid is None
            ):  # if this is the initial call for deployed chat model
                if version == "deploy":
                    for version in chat_model_versions:
                        if version["is_published"] is True:
                            version["ratio"] = 1.0
                    selected_version = select_version_by_ratio(chat_model_versions)
                else:
                    selected_version = chat_model_versions[0]
            else:
                selected_version = chat_model_versions[0]

            instruction: str = selected_version["system_prompt"]

            version_detail = {
                "model": selected_version["model"],
                "uuid": selected_version["uuid"],
            }
            if session_uuid:
                chat_logs: List[Dict] = res_data["chat_logs"]
                chat_logs = [{"role": "system", "content": instruction}] + chat_logs
            else:
                chat_logs = []

            # delete columns which value is None in each chat log
            for chat_log in chat_logs:
                for key in list(chat_log.keys()):
                    if chat_log[key] is None:
                        del chat_log[key]

            return instruction, version_detail, chat_logs

    # @staticmethod
    # async def fetch_chat_log(
    #     session_uuid: str,
    #     version: Optional[Union[str, int]] = "deploy",
    # ) -> List[Dict[str, Any]]:
    #     """fetch conversation log for session_uuid and version detail

    #     Args:
    #         session_uuid (str): session_uuid

    #     Returns:
    #         List[Dict[str, Any]] : list of conversation log
    #     """
    #     config = read_config()
    #     if "connection" in config and config["connection"]["initializing"] == True:
    #         return []
    #     elif "connection" in config and config["connection"]["reloading"] == True:
    #         return []
    #     else:
    #         try:
    #             res_data = await AsyncAPIClient.execute(
    #                 method="GET",
    #                 path="/fetch_chat_logs",
    #                 params={"session_uuid": session_uuid},
    #                 use_cli_key=False,
    #             )
    #             res_data = res_data.json()
    #         except Exception as e:
    #             raise e

    #         # filter out unnecessary data
    #         res_data = [
    #             {
    #                 "role": message["role"],
    #                 "content": message["content"],
    #                 "function_call": message["function_call"],
    #             }
    #             for message in res_data["chat_logs"]
    #         ]
    #         return res_data
