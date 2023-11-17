import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any, Optional
from litellm import ModelResponse

from ..constants import function_shemas
from promptmodel.llms.llm import LLM, ParseResult
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.types.response import (
    LLMResponse,
    LLMStreamResponse,
    Message,
    Delta,
    Choices,
    Usage,
    FunctionCall,
    ChoiceDeltaFunctionCall,
    StreamingChoices,
)
from promptmodel.types.enums import ParsingType


def test_parse_output_pattern_error_cases(mocker):
    llm = LLM()
    res: ParseResult = llm.__parse_output_pattern__(parsing_type=None)
    assert res.parsed_outputs == {}
    assert res.error == False
    assert res.error_log is None

    res: ParseResult = llm.__parse_output_pattern__(
        parsing_type=ParsingType.COLON.value, raw_output=None
    )
    assert res.parsed_outputs == {}
    assert res.error == True
    assert res.error_log is not None

    res: ParseResult = llm.__parse_output_pattern__(
        parsing_type=ParsingType.HTML.value,
        raw_output="<response type=int>hello</response>",
    )
    assert res.parsed_outputs == {}
    assert res.error == True
    assert res.error_log is not None


def test_run_error_cases(mocker):
    llm = LLM()
    mocker.patch("promptmodel.llms.llm.completion", return_value={})
    res: LLMResponse = llm.run(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert res.api_response is not None
    assert res.error == True

    mocker.patch("promptmodel.llms.llm.completion", side_effect=Exception("test"))
    res: LLMResponse = llm.run(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert res.api_response is None
    assert res.error == True


@pytest.mark.asyncio
async def test_arun_error_cases(mocker):
    llm = LLM()
    mocker.patch("promptmodel.llms.llm.acompletion", return_value={})
    res: LLMResponse = await llm.arun(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert res.api_response is not None
    assert res.error == True

    mocker.patch("promptmodel.llms.llm.acompletion", side_effect=Exception("test"))
    res: LLMResponse = await llm.arun(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert res.api_response is None
    assert res.error == True


def test_stream_error_cases(mocker):
    llm = LLM()

    mocker.patch("promptmodel.llms.llm.completion", side_effect=Exception("test"))
    res: LLMResponse = llm.stream(
        messages=[{"role": "user", "content": "hello"}],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert len(results) == 1
    assert results[0].api_response is None
    assert results[0].error == True


@pytest.mark.asyncio
async def test_astream_error_cases(mocker):
    llm = LLM()

    mocker.patch("promptmodel.llms.llm.acompletion", side_effect=Exception("test"))
    res: LLMResponse = llm.astream(
        messages=[{"role": "user", "content": "hello"}],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)

    assert len(results) == 1
    assert results[0].api_response is None
    assert results[0].error == True


def string_to_generator(input_string: str):
    def generator_format(response: str):
        model_response = ModelResponse(stream=True)
        model_response.choices[0] = StreamingChoices(
            **{
                "delta": Delta(**{"content": response}),
                "finish_reason": None,
            }
        )
        return model_response

    def generator():
        for char in input_string:
            yield generator_format(char)
        model_response = ModelResponse(stream=True)
        model_response.choices[0] = StreamingChoices(
            **{"delta": Delta(**{"content": None}), "finish_reason": "stop"}
        )
        yield model_response

    return generator()


async def string_to_agenerator(input_string: str):
    def generator_format(response: str):
        model_response = ModelResponse(stream=True)
        model_response.choices[0] = StreamingChoices(
            **{
                "delta": Delta(**{"content": response}),
                "finish_reason": None,
            }
        )
        return model_response

    async def agenerator():
        for char in input_string:
            yield generator_format(char)
        model_response = ModelResponse(stream=True)
        model_response.choices[0] = StreamingChoices(
            **{"delta": Delta(**{"content": None}), "finish_reason": "stop"}
        )
        yield model_response

    return agenerator()


def test_stream_and_parse_error_cases(mocker):
    llm = LLM()

    # error case - colon type
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.COLON.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert results[0].api_response is None
    assert results[0].error == True
    assert results[0].error_log == "Cannot stream colon type"

    # error case - side effect
    mocker.patch(
        "promptmodel.llms.llm.completion",
        side_effect=Exception("test"),
    )
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    assert len([result for result in results if result.api_response is not None]) == 0

    # # html success case
    # mocker.patch(
    #     "promptmodel.llms.llm.completion",
    #     return_value=string_to_generator("<response type=str>hello</response>"),
    # )
    # res: LLMResponse = llm.stream(
    #     messages=[{"role": "user", "content": "hello"}],
    #     parsing_type=ParsingType.HTML.value,
    #     output_keys=["response"],
    # )
    # results: List[LLMStreamResponse] = []
    # for chunk in res:
    #     results.append(chunk)
    # assert len([result for result in results if result.error == True]) == 0

    # html failed case - error in parsing
    mocker.patch(
        "promptmodel.llms.llm.completion",
        return_value=string_to_generator("<response type=str>hello<response>"),
    )
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0

    # html failed case - error in key matching
    mocker.patch(
        "promptmodel.llms.llm.completion",
        return_value=string_to_generator("<respond type=str>hello</respond>"),
    )
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    error_log = [result.error_log for result in results if result.error == True][0]
    assert error_log == "Output keys do not match with parsed output keys"

    # html failed case - Having functions as input case
    mocker.patch(
        "promptmodel.llms.llm.completion",
        return_value=string_to_generator("<respond type=str>hello</respond>"),
    )
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
        functions=function_shemas,
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
        # print(chunk.__dict__)
    assert len([result for result in results if result.error == True]) > 0
    error_log = [result.error_log for result in results if result.error == True][0]
    print([result.error_log for result in results if result.error == True])
    assert error_log == "Output keys do not match with parsed output keys"

    # Double Bracket failed case - error in parsing
    mocker.patch(
        "promptmodel.llms.llm.completion",
        return_value=string_to_generator("[[response type=str]]hello[[response]]"),
    )
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.DOUBLE_SQUARE_BRACKET.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0

    # Double Bracket failed case - error in key matching
    mocker.patch(
        "promptmodel.llms.llm.completion",
        return_value=string_to_generator("[[respond type=str]]hello[[/respond]]"),
    )
    res: LLMResponse = llm.stream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.DOUBLE_SQUARE_BRACKET.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    error_log = [result.error_log for result in results if result.error == True][0]
    assert error_log == "Output keys do not match with parsed output keys"


@pytest.mark.asyncio
async def test_astream_and_parse_error_cases(mocker):
    llm = LLM()

    # error case - colon type
    res: AsyncGenerator[LLMResponse, None] = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.COLON.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert results[0].api_response is None
    assert results[0].error == True
    assert results[0].error_log == "Cannot stream colon type"

    # error case - side effect
    mocker.patch(
        "promptmodel.llms.llm.acompletion",
        side_effect=Exception("test"),
    )
    res: AsyncGenerator[LLMResponse, None] = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    assert len([result for result in results if result.api_response is not None]) == 0

    # # html success case
    # mocker.patch(
    #     "promptmodel.llms.llm.completion",
    #     return_value=string_to_generator("<response type=str>hello</response>"),
    # )
    # res: LLMResponse = llm.stream(
    #     messages=[{"role": "user", "content": "hello"}],
    #     parsing_type=ParsingType.HTML.value,
    #     output_keys=["response"],
    # )
    # results: List[LLMStreamResponse] = []
    # for chunk in res:
    #     results.append(chunk)
    # assert len([result for result in results if result.error == True]) == 0

    # html failed case - error in parsing
    mocker.patch(
        "promptmodel.llms.llm.acompletion",
        return_value=await string_to_agenerator("<response type=str>hello<response>"),
    )
    res: AsyncGenerator[LLMResponse, None] = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0

    # html failed case - error in key matching
    mocker.patch(
        "promptmodel.llms.llm.acompletion",
        return_value=await string_to_agenerator("<respond type=str>hello</respond>"),
    )
    res: AsyncGenerator[LLMResponse, None] = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    error_log = [result.error_log for result in results if result.error == True][0]
    assert error_log == "Output keys do not match with parsed output keys"

    # html failed case - Having functions as input case
    mocker.patch(
        "promptmodel.llms.llm.completion",
        return_value=await string_to_agenerator("<respond type=str>hello</respond>"),
    )
    res: LLMResponse = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.HTML.value,
        output_keys=["response"],
        functions=function_shemas,
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    error_log = [result.error_log for result in results if result.error == True][0]
    assert error_log == "Output keys do not match with parsed output keys"

    # Double Bracket failed case - error in parsing
    mocker.patch(
        "promptmodel.llms.llm.acompletion",
        return_value=await string_to_agenerator(
            "[[response type=str]]hello[[response]]"
        ),
    )
    res: AsyncGenerator[LLMResponse, None] = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.DOUBLE_SQUARE_BRACKET.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0

    # Double Bracket failed case - error in key matching
    mocker.patch(
        "promptmodel.llms.llm.acompletion",
        return_value=await string_to_agenerator(
            "[[respond type=str]]hello[[/respond]]"
        ),
    )
    res: AsyncGenerator[LLMResponse, None] = llm.astream_and_parse(
        messages=[{"role": "user", "content": "hello"}],
        parsing_type=ParsingType.DOUBLE_SQUARE_BRACKET.value,
        output_keys=["response"],
    )
    results: List[LLMStreamResponse] = []
    async for chunk in res:
        results.append(chunk)
    assert len([result for result in results if result.error == True]) > 0
    error_log = [result.error_log for result in results if result.error == True][0]
    assert error_log == "Output keys do not match with parsed output keys"
