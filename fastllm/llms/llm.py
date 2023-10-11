"""Base module for interacting with OpenAI's Language Model API."""
"""Base module for interacting with OpenAI's Language Model API."""
import re
import os
import json
from typing import Any, AsyncGenerator, List, Dict, Optional, Union, Generator
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from ..utils import logger
from litellm import completion

load_dotenv()


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class OpenAIMessage(BaseModel):
    role: str
    content: str


class LLM:
    def __init__(self):
        self._model: str

    @classmethod
    def __parse_output__(cls, raw_output: str, key: str) -> Union[str, None]:
        """Parse value for key from raw output."""
        # capitalized_key = key[0].upper() + key[1:]  # capitalize the first letter of the key
        pattern = r"\[\[{key}(\s*\(.+\))?\sstart\]\](.*?)\[\[{key}(\s*\(.+\))?\send\]\]".format(
            # key=capitalized_key
            key=key
        )
        results = re.findall(pattern, raw_output, flags=re.DOTALL)
        results = [result[1] for result in results]
        if results:
            if len(results) > 1:
                raise ValueError("Multiple Matches")
            return results[0].strip()
        else:
            return None

    def __validate_openai_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        return [OpenAIMessage(**message) for message in messages]

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ):
        """Return the response from openai chat completion."""
        _model = model or self._model
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        ).choices[0]["message"]["content"]

        return response

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ):
        """Return the response from openai chat completion."""
        _model = model or self._model
        response = await completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        )
        res = response.choices[0]["message"]["content"]

        return res

    def stream(
        self,
        messages: List[Dict[str, str]],  # input
        model: Optional[str] = None,
    ):
        """Stream openai chat completion."""
        _model = model or self._model
        # load_prompt()
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        ).choices[0]["message"]["content"]
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["content"]

    def generate_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """Parse and return output from openai chat completion."""
        _model = model or self._model
        raw_output = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        ).choices[0]["message"]["content"]

        parsed_output = {}
        for key in output_keys:
            parsed_output[key] = self.__parse_output__(raw_output, key)

        return parsed_output

    def stream_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[Dict[str, str], None, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        response = completion(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )

        cache = ""
        for chunk in response:
            pause_stream = False
            if "content" in chunk["choices"][0]["delta"]:
                stream_value = chunk["choices"][0]["delta"]["content"]
                raw_output += stream_value  # 지금까지 생성된 누적 output
                pattern = (
                    r"\[\[.*?(\s*\(.+\))?\sstart\]\](.*?)\[\[.*?(\s*\(.+\))?\send\]\]"
                )
                stripped_output = re.sub(
                    pattern, "", raw_output, flags=re.DOTALL
                )  # 누적 output에서 [key start] ~ [key end] 부분을 제거한 output
                streaming_key = re.findall(
                    r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                    stripped_output,
                    flags=re.DOTALL,  # stripped output에서 [key start] 부분을 찾음
                )
                if not streaming_key:  # 아직 output value를 streaming 중이 아님
                    continue

                if len(streaming_key) > 1:
                    raise ValueError("Multiple Matches")
                # key = streaming_key[0].lower()
                key = streaming_key[0]
                if key not in output_keys:  # 미리 정해둔 output key가 아님
                    continue
                if stream_value.find("]") != -1 or "[" in re.sub(
                    r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                    "",
                    stripped_output,
                    flags=re.DOTALL,
                ):  # 현재 stream 중인 output이 [[key end]] 부분일 경우에는 pause_stream을 True로 설정
                    if stream_value.find("[") != -1:
                        if cache.find("[[") != -1:
                            pause_stream = True
                        else:
                            cache += "["
                    pause_stream = True
                if pause_stream:
                    if stream_value.find("]") != -1:
                        cache = ""
                        pause_stream = False
                    continue

                yield {key: stream_value}

    async def agenerate_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate openai chat completion asynchronously, and parse the output.
        Example prompt is as follows:
        -----
        Given a topic, you are required to generate a story.
        You must follow the provided output format.

        Topic:
        {topic}

        Output format:
        [Story start]
        ...
        [Story end]

        Now generate the output:
        """
        _model = model or self._model
        result = await openai.ChatCompletion.acreate(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
        )
        raw_output = result.choices[0]["message"]["content"]
        logger.debug(f"Output:\n{raw_output}")
        parsed_output = {}
        for key in output_keys:
            output = self.__parse_output__(raw_output, key)
            if output:
                parsed_output[key] = output

        return parsed_output

    def generate_and_parse_function_call(
        self,
        messages: List[Dict[str, str]],
        function_list: [],
        model: Optional[str] = "gpt-3.5-turbo-0613",
    ) -> Generator[str, None, None]:
        """
        Parse by function call arguments
        """
        response = completion(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            functions=function_list,
            function_call="auto",
        )
        print(response)
        function_args = response["choices"][0]["message"]["function_call"]["arguments"]
        # make function_args to dict
        function_args = function_args.replace("'", '"')
        function_args = json.loads(function_args)
        return function_args

    async def agenerate_and_parse_function_call(
        self,
        messages: List[Dict[str, str]],
        function_list: [],
        model: Optional[str] = "gpt-3.5-turbo-0613",
    ) -> Generator[str, None, None]:
        """
        Parse by function call arguments
        """
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            functions=function_list,
            function_call="auto",
        )
        print(response)
        function_args = response["choices"][0]["message"]["function_call"]["arguments"]
        # make function_args to dict
        function_args = function_args.replace("'", '"')
        function_args = json.loads(function_args)
        return function_args

    async def astream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Generator[Dict[str, str], None, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        response = await openai.ChatCompletion.acreate(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )
        async for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["content"]

    async def astream_and_parse(
        self,
        messages: List[Dict[str, str]],
        output_keys: List[str],
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        response = await openai.ChatCompletion.acreate(
            model=_model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
        )

        async for chunk in response:
            pause_stream = False
            if "content" in chunk["choices"][0]["delta"]:
                stream_value = chunk["choices"][0]["delta"]["content"]
                raw_output += stream_value  # 지금까지 생성된 누적 output
                pattern = (
                    r"\[\[.*?(\s*\(.+\))?\sstart\]\](.*?)\[\[.*?(\s*\(.+\))?\send\]\]"
                )
                stripped_output = re.sub(
                    pattern, "", raw_output, flags=re.DOTALL
                )  # 누적 output에서 [key start] ~ [key end] 부분을 제거한 output
                streaming_key = re.findall(
                    r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                    stripped_output,
                    flags=re.DOTALL,  # stripped output에서 [key start] 부분을 찾음
                )
                if not streaming_key:  # 아직 output value를 streaming 중이 아님
                    continue

                if len(streaming_key) > 1:
                    raise ValueError("Multiple Matches")
                # key = streaming_key[0].lower()
                key = streaming_key[0]
                if key not in output_keys:  # 미리 정해둔 output key가 아님
                    continue
                if stream_value.find("]") != -1 or "[" in re.sub(
                    r"\[\[(.*?)(?:\s*\(.+\))?\sstart\]\]",
                    "",
                    stripped_output.split(f"[[{key} start]]")[-1],
                    flags=re.DOTALL,
                ):  # 현재 stream 중인 output이 [[key end]] 부분일 경우에는 pause_stream을 True로 설정
                    if stream_value.find("[") != -1:
                        if cache.find("[[") != -1:
                            logger.info("[[ in cache")
                            pause_stream = True
                        else:
                            cache += "["
                    pause_stream = True
                if not pause_stream:
                    yield {key: stream_value}
                elif stream_value.find("]") != -1:
                    # Current stream_value (that includes ]) isn't yielded, but the next stream_values will be yielded.
                    cache = ""
                    pause_stream = False

    async def aget_embedding(self, context: str) -> List[float]:
        """
        Return the embedding of the context.
        """
        context = context.replace("\n", " ")
        response = await openai.Embedding.acreate(
            input=[context], model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    async def astream_and_parse_function_call(
        self,
        messages: List[Dict[str, str]],
        function_list: [],
        output_key: str,
        model: Optional[str] = "gpt-3.5-turbo-0613",
    ) -> Generator[str, None, None]:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                message.model_dump()
                for message in self.__validate_openai_messages(messages)
            ],
            functions=function_list,
            function_call="auto",
            stream=True,
        )
        function_args = ""
        start_to_stream = False
        async for chunk in response:
            if "function_call" in chunk["choices"][0]["delta"]:
                if chunk["choices"][0]["delta"]["function_call"]:
                    # if "name" in chunk["choices"][0]["delta"]["function_call"]:
                    #     function_name = chunk["choices"][0]["delta"]["function_call"][
                    #         "name"
                    #     ]
                    function_args += chunk["choices"][0]["delta"]["function_call"][
                        "arguments"
                    ]
                    if f'"{output_key}":' in function_args:
                        if not start_to_stream:
                            start_to_stream = True
                            # yield function_args without output_key
                            yield {
                                output_key: function_args.replace(
                                    f'"{output_key}":', ""
                                )
                            }
                        else:
                            yield {
                                output_key: chunk["choices"][0]["delta"][
                                    "function_call"
                                ]["arguments"]
                            }
            else:
                continue
