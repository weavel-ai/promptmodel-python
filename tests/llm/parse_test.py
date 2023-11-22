import pytest
import re
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any
from litellm import ModelResponse

from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import (
    LLMResponse,
    LLMStreamResponse,
    Choices,
    StreamingChoices,
    Delta,
    Message,
    FunctionCall,
)
from promptmodel.types.enums import ParsingPattern, ParsingType, get_pattern_by_type


def generator_format(response: str):
    model_response = ModelResponse(stream=True)
    model_response.choices[0] = StreamingChoices(
        **{
            "delta": Delta(**{"content": response}),
            "finish_reason": None,
        }
    )
    return model_response


def string_to_generator(input_string: str):
    def generator():
        for char in input_string:
            yield generator_format(char)

    return generator()


import random


def random_chunk_generator(input_string: str):
    def generator():
        start = 0
        while start < len(input_string):
            chunk_size = random.randint(1, 3)
            end = min(start + chunk_size, len(input_string))
            yield generator_format(input_string[start:end])
            start = end

    return generator()


async def string_to_agenerator(input_string: str):
    async def agenerator():
        for char in input_string:
            yield generator_format(char)

    return agenerator()


import random


async def random_chunk_agenerator(input_string: str):
    async def agenerator():
        start = 0
        while start < len(input_string):
            chunk_size = random.randint(1, 3)
            end = min(start + chunk_size, len(input_string))
            yield generator_format(input_string[start:end])
            start = end

    return agenerator()


class ResultClass:
    raw_output: str
    parsed_outputs: Dict[str, str]
    parse_success: bool

    def __init__(self, raw_output, parsed_outputs, parse_success):
        self.raw_output = raw_output
        self.parsed_outputs = parsed_outputs
        self.parse_success = parse_success


bracket_allow_example_gen = string_to_generator("[key type=str]ab[/key]")
bracket_allow_result = ResultClass("[key type=str]ab[/key]", {"key": "ab"}, True)

bracket_forbidden_example_gen = string_to_generator("[key type=str]ab[/not key]")
bracket_forbidden_result = ResultClass("[key type=str]ab[/not key]", {}, False)
bracket_forbidden_stream_result = ResultClass(
    "[key type=str]ab[/not key]", {"key": "ab[/not key]"}, False
)

double_allow_example_gen = string_to_generator(
    "[[key type=str]]abc[not key]xy[/not key]de[[/key]]"
)
double_allow_result = ResultClass(
    "[[key type=str]]abc[not key]xy[/not key]de[[/key]]",
    {"key": "abc[not key]xy[/not key]de"},
    True,
)

double_forbidden_example_gen = string_to_generator(
    "[[key type=str]]abc[not key]xy[/not key]de[[/not key]]"
)
double_forbidden_result = ResultClass(
    "[[key type=str]]abc[not key]xy[/not key]de[[/not key]]", {}, False
)
double_forbidden_stream_result = ResultClass(
    "[[key type=str]]abc[not key]xy[/not key]de[[/not key]]",
    {"key": "abc[not key]xy[/not key]de"},
    False,
)


def generator_processor(generator: Generator):
    raw_output = ""
    parsed_outputs = {}
    parse_success = True
    for chunk in generator:
        # print(chunk)
        if type(chunk) == str:
            raw_output += chunk
        elif type(chunk) == dict:
            for key, value in chunk.items():
                if key in parsed_outputs:
                    parsed_outputs[key] += value
                else:
                    parsed_outputs[key] = value
        elif type(chunk) == bool:
            parse_success = chunk
        elif type(chunk) == LLMResponse:
            if chunk.raw_output:
                raw_output += chunk.raw_output
            if chunk.parsed_outputs:
                for key, value in chunk.parsed_outputs.items():
                    if key in parsed_outputs:
                        parsed_outputs[key] += value
                    else:
                        parsed_outputs[key] = value
            if parse_success is True and chunk.error is True:
                parse_success = False
        elif type(chunk) == LLMStreamResponse:
            if chunk.raw_output:
                raw_output += chunk.raw_output
            if chunk.parsed_outputs:
                for key, value in chunk.parsed_outputs.items():
                    if key in parsed_outputs:
                        parsed_outputs[key] += value
                    else:
                        parsed_outputs[key] = value
            if parse_success is True and chunk.error is True:
                parse_success = False

    return raw_output, parsed_outputs, parse_success


async def agenerator_processor(generator: AsyncGenerator):
    raw_output = ""
    parsed_outputs = {}
    parse_success = True
    async for chunk in generator:
        # print(chunk)
        if type(chunk) == str:
            raw_output += chunk
        elif type(chunk) == dict:
            for key, value in chunk.items():
                if key in parsed_outputs:
                    parsed_outputs[key] += value
                else:
                    parsed_outputs[key] = value
        elif type(chunk) == bool:
            parse_success = chunk
        elif type(chunk) == LLMResponse:
            if chunk.raw_output:
                raw_output += chunk.raw_output
            if chunk.parsed_outputs:
                for key, value in chunk.parsed_outputs.items():
                    if key in parsed_outputs:
                        parsed_outputs[key] += value
                    else:
                        parsed_outputs[key] = value
            if parse_success is True and chunk.error is True:
                parse_success = False
        elif type(chunk) == LLMStreamResponse:
            if chunk.raw_output:
                raw_output += chunk.raw_output
            if chunk.parsed_outputs:
                for key, value in chunk.parsed_outputs.items():
                    if key in parsed_outputs:
                        parsed_outputs[key] += value
                    else:
                        parsed_outputs[key] = value
            if parse_success is True and chunk.error is True:
                parse_success = False

    return raw_output, parsed_outputs, parse_success


def test_sp_generator(mocker):
    llm = LLM()

    raw_output, parsed_outputs, parse_success = generator_processor(
        llm.__single_type_sp_generator__(
            None,
            bracket_allow_example_gen,
            ParsingType.SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)

    assert (
        raw_output == bracket_allow_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == bracket_allow_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == bracket_allow_result.parse_success
    ), f"parse success mismatch : {parse_success}"

    raw_output, parsed_outputs, parse_success = generator_processor(
        llm.__single_type_sp_generator__(
            None,
            bracket_forbidden_example_gen,
            ParsingType.SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)
    assert (
        raw_output == bracket_forbidden_stream_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == bracket_forbidden_stream_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == bracket_forbidden_stream_result.parse_success
    ), f"parse success mismatch : {parse_success}"
    # test Random Generator 5 times
    for i in range(5):
        raw_output, parsed_outputs, parse_success = generator_processor(
            llm.__single_type_sp_generator__(
                None,
                random_chunk_generator("[key type=str]ab[/key]"),
                ParsingType.SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)

        assert (
            raw_output == bracket_allow_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == bracket_allow_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == bracket_allow_result.parse_success
        ), f"parse success mismatch : {parse_success}"

        raw_output, parsed_outputs, parse_success = generator_processor(
            llm.__single_type_sp_generator__(
                None,
                random_chunk_generator("[key type=str]ab[/not key]"),
                ParsingType.SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)
        assert (
            raw_output == bracket_forbidden_stream_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == bracket_forbidden_stream_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == bracket_forbidden_stream_result.parse_success
        ), f"parse success mismatch : {parse_success}"


def test_dp_generator(mocker):
    llm = LLM()
    raw_output, parsed_outputs, parse_success = generator_processor(
        llm.__double_type_sp_generator__(
            None,
            double_allow_example_gen,
            ParsingType.DOUBLE_SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)

    assert (
        raw_output == double_allow_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == double_allow_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == double_allow_result.parse_success
    ), f"parse success mismatch : {parse_success}"

    raw_output, parsed_outputs, parse_success = generator_processor(
        llm.__double_type_sp_generator__(
            None,
            double_forbidden_example_gen,
            ParsingType.DOUBLE_SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)
    assert (
        raw_output == double_forbidden_stream_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == double_forbidden_stream_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == double_forbidden_stream_result.parse_success
    ), f"parse success mismatch : {parse_success}"

    # test random generator 5 times
    for i in range(5):
        raw_output, parsed_outputs, parse_success = generator_processor(
            llm.__double_type_sp_generator__(
                None,
                random_chunk_generator(
                    "[[key type=str]]abc[not key]xy[/not key]de[[/key]]"
                ),
                ParsingType.DOUBLE_SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)

        assert (
            raw_output == double_allow_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == double_allow_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == double_allow_result.parse_success
        ), f"parse success mismatch : {parse_success}"

        raw_output, parsed_outputs, parse_success = generator_processor(
            llm.__double_type_sp_generator__(
                None,
                random_chunk_generator(
                    "[[key type=str]]abc[not key]xy[/not key]de[[/not key]]"
                ),
                ParsingType.DOUBLE_SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)
        assert (
            raw_output == double_forbidden_stream_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == double_forbidden_stream_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == double_forbidden_stream_result.parse_success
        ), f"parse success mismatch : {parse_success}"


@pytest.mark.asyncio
async def test_sp_agenerator(mocker):
    llm = LLM()
    bracket_allow_example_agen = await string_to_agenerator("[key type=str]ab[/key]")
    bracket_forbidden_example_agen = await string_to_agenerator(
        "[key type=str]ab[/not key]"
    )

    raw_output, parsed_outputs, parse_success = await agenerator_processor(
        llm.__single_type_sp_agenerator__(
            None,
            bracket_allow_example_agen,
            ParsingType.SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)

    assert (
        raw_output == bracket_allow_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == bracket_allow_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == bracket_allow_result.parse_success
    ), f"parse success mismatch : {parse_success}"

    raw_output, parsed_outputs, parse_success = await agenerator_processor(
        llm.__single_type_sp_agenerator__(
            None,
            bracket_forbidden_example_agen,
            ParsingType.SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)
    assert (
        raw_output == bracket_forbidden_stream_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == bracket_forbidden_stream_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == bracket_forbidden_stream_result.parse_success
    ), f"parse success mismatch : {parse_success}"
    # test Random Generator 5 times
    for i in range(5):
        raw_output, parsed_outputs, parse_success = await agenerator_processor(
            llm.__single_type_sp_agenerator__(
                None,
                await random_chunk_agenerator("[key type=str]ab[/key]"),
                ParsingType.SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)

        assert (
            raw_output == bracket_allow_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == bracket_allow_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == bracket_allow_result.parse_success
        ), f"parse success mismatch : {parse_success}"

        raw_output, parsed_outputs, parse_success = await agenerator_processor(
            llm.__single_type_sp_agenerator__(
                None,
                await random_chunk_agenerator("[key type=str]ab[/not key]"),
                ParsingType.SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)
        assert (
            raw_output == bracket_forbidden_stream_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == bracket_forbidden_stream_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == bracket_forbidden_stream_result.parse_success
        ), f"parse success mismatch : {parse_success}"


@pytest.mark.asyncio
async def test_dp_agenerator(mocker):
    llm = LLM()
    double_allow_example_agen = await string_to_agenerator(
        "[[key type=str]]abc[not key]xy[/not key]de[[/key]]"
    )
    double_forbidden_example_agen = await string_to_agenerator(
        "[[key type=str]]abc[not key]xy[/not key]de[[/not key]]"
    )

    raw_output, parsed_outputs, parse_success = await agenerator_processor(
        llm.__double_type_sp_agenerator__(
            None,
            double_allow_example_agen,
            ParsingType.DOUBLE_SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)

    assert (
        raw_output == double_allow_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == double_allow_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == double_allow_result.parse_success
    ), f"parse success mismatch : {parse_success}"

    raw_output, parsed_outputs, parse_success = await agenerator_processor(
        llm.__double_type_sp_agenerator__(
            None,
            double_forbidden_example_agen,
            ParsingType.DOUBLE_SQUARE_BRACKET.value,
            datetime.now(),
        )
    )
    print(raw_output, parsed_outputs, parse_success)
    assert (
        raw_output == double_forbidden_stream_result.raw_output
    ), f"raw output error : {raw_output}"
    assert (
        parsed_outputs == double_forbidden_stream_result.parsed_outputs
    ), f"parsed output error : {parsed_outputs}"
    assert (
        parse_success == double_forbidden_stream_result.parse_success
    ), f"parse success mismatch : {parse_success}"

    # test random generator 5 times
    for i in range(5):
        raw_output, parsed_outputs, parse_success = await agenerator_processor(
            llm.__double_type_sp_agenerator__(
                None,
                await random_chunk_agenerator(
                    "[[key type=str]]abc[not key]xy[/not key]de[[/key]]"
                ),
                ParsingType.DOUBLE_SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)

        assert (
            raw_output == double_allow_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == double_allow_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == double_allow_result.parse_success
        ), f"parse success mismatch : {parse_success}"

        raw_output, parsed_outputs, parse_success = await agenerator_processor(
            llm.__double_type_sp_agenerator__(
                None,
                await random_chunk_agenerator(
                    "[[key type=str]]abc[not key]xy[/not key]de[[/not key]]"
                ),
                ParsingType.DOUBLE_SQUARE_BRACKET.value,
                datetime.now(),
            )
        )
        print(raw_output, parsed_outputs, parse_success)
        assert (
            raw_output == double_forbidden_stream_result.raw_output
        ), f"raw output error : {raw_output}"
        assert (
            parsed_outputs == double_forbidden_stream_result.parsed_outputs
        ), f"parsed output error : {parsed_outputs}"
        assert (
            parse_success == double_forbidden_stream_result.parse_success
        ), f"parse success mismatch : {parse_success}"


@pytest.mark.asyncio
async def test_run_and_parsing(mocker):
    llm = LLM()

    mock_response = ModelResponse()
    mock_response.choices = [
        Choices(
            **{
                "index": 0,
                "message": Message(
                    **{
                        "role": "assistant",
                        "content": "[key type=str]ab[/key]",
                    }
                ),
                "finish_reason": "function_call",
            }
        )
    ]
    mock_completion = mocker.patch(
        "promptmodel.llms.llm.completion", return_value=mock_response
    )

    # success case
    res: LLMResponse = llm.run_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.parsed_outputs == {
        "key": "ab"
    }, f"parsed output error : {res.parsed_outputs}"

    # failed case
    res: LLMResponse = llm.run_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key", "key2"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.parsed_outputs == {
        "key": "ab"
    }, f"parsed output error : {res.parsed_outputs}"
    assert res.error == True, f"error mismatch : {res.error}"

    # error case, raise in __parse_output_pattern__
    mock_completion = mocker.patch(
        "promptmodel.llms.llm.completion", return_value=mock_response
    )
    with mocker.patch(
        "promptmodel.llms.llm.LLM.__parse_output_pattern__",
        side_effect=Exception("test"),
    ):
        res: LLMResponse = llm.run_and_parse(
            messages=[
                {"role": "user", "content": "What is the weather like in Boston?"}
            ],
            parsing_type=ParsingType.SQUARE_BRACKET.value,
            functions=[],
            output_keys=["key"],
            model="gpt-3.5-turbo",
        )
        mock_completion.assert_called_once()
        mock_completion.reset_mock()
        # should have api response
        assert (
            res.api_response is not None
        ), f"api response mismatch : {res.api_response}"
        assert res.error == True, f"error mismatch : {res.error}"

    # success case with function call
    mock_response.choices[0].message.function_call = FunctionCall(
        **{
            "name": "get_current_weather",
            "arguments": '{\n  "location": "Boston, MA"\n}',
        }
    )
    res: LLMResponse = llm.run_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert (
        res.parsed_outputs is None
    ), f"If function call, do not parse but : {res.parsed_outputs}"
    assert res.function_call == FunctionCall(
        **{
            "name": "get_current_weather",
            "arguments": '{\n  "location": "Boston, MA"\n}',
        }
    ), f"function call mismatch : {res.function_call}"

    # success case because function_call, did not parse
    mocker.patch(
        "promptmodel.llms.llm.LLM.__parse_output_pattern__",
        side_effect=Exception("test"),
    )
    res: LLMResponse = llm.run_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.api_response is not None, f"api response mismatch : {res.api_response}"
    assert res.error == False, f"error mismatch : {res.error}"

    # error case, raise error in completion
    mock_completion = mocker.patch(
        "promptmodel.llms.llm.completion",
        side_effect=Exception("test"),
        return_value=None,
    )
    res: LLMResponse = llm.run_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.api_response is None, f"api response mismatch : {res.api_response}"
    assert res.error == True, f"error mismatch : {res.error}"


@pytest.mark.asyncio
async def test_arun_and_parsing(mocker):
    llm = LLM()

    mock_response = ModelResponse()
    mock_response.choices = [
        Choices(
            **{
                "index": 0,
                "message": Message(
                    **{
                        "role": "assistant",
                        "content": "[key type=str]ab[/key]",
                    }
                ),
                "finish_reason": "function_call",
            }
        )
    ]
    mock_completion = mocker.patch(
        "promptmodel.llms.llm.acompletion", return_value=mock_response
    )

    # success case
    res: LLMResponse = await llm.arun_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.parsed_outputs == {
        "key": "ab"
    }, f"parsed output error : {res.parsed_outputs}"

    # failed case
    res: LLMResponse = await llm.arun_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key", "key2"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.parsed_outputs == {
        "key": "ab"
    }, f"parsed output error : {res.parsed_outputs}"
    assert res.error == True, f"error mismatch : {res.error}"

    # error case, raise in __parse_output_pattern__
    mock_completion = mocker.patch(
        "promptmodel.llms.llm.acompletion", return_value=mock_response
    )
    with mocker.patch(
        "promptmodel.llms.llm.LLM.__parse_output_pattern__",
        side_effect=Exception("test"),
    ):
        res: LLMResponse = await llm.arun_and_parse(
            messages=[
                {"role": "user", "content": "What is the weather like in Boston?"}
            ],
            parsing_type=ParsingType.SQUARE_BRACKET.value,
            functions=[],
            output_keys=["key"],
            model="gpt-3.5-turbo",
        )
        mock_completion.assert_called_once()
        mock_completion.reset_mock()
        # should have api response
        assert (
            res.api_response is not None
        ), f"api response mismatch : {res.api_response}"
        assert res.error == True, f"error mismatch : {res.error}"

    # success case with function call
    mock_response.choices[0].message.function_call = FunctionCall(
        **{
            "name": "get_current_weather",
            "arguments": '{\n  "location": "Boston, MA"\n}',
        }
    )
    res: LLMResponse = await llm.arun_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert (
        res.parsed_outputs is None
    ), f"If function call, do not parse but : {res.parsed_outputs}"
    assert res.function_call == FunctionCall(
        **{
            "name": "get_current_weather",
            "arguments": '{\n  "location": "Boston, MA"\n}',
        }
    ), f"function call mismatch : {res.function_call}"

    # success case because function_call, did not parse
    mocker.patch(
        "promptmodel.llms.llm.LLM.__parse_output_pattern__",
        side_effect=Exception("test"),
    )
    res: LLMResponse = await llm.arun_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.api_response is not None, f"api response mismatch : {res.api_response}"
    assert res.error == False, f"error mismatch : {res.error}"

    # error case, raise error in completion
    mock_completion = mocker.patch(
        "promptmodel.llms.llm.acompletion",
        side_effect=Exception("test"),
        return_value=None,
    )
    res: LLMResponse = await llm.arun_and_parse(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        parsing_type=ParsingType.SQUARE_BRACKET.value,
        functions=[],
        output_keys=["key"],
        model="gpt-3.5-turbo",
    )
    mock_completion.assert_called_once()
    mock_completion.reset_mock()
    assert res.api_response is None, f"api response mismatch : {res.api_response}"
    assert res.error == True, f"error mismatch : {res.error}"
