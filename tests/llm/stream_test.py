import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from typing import Generator, AsyncGenerator, Dict, List, Any
from litellm import ModelResponse

from promptmodel.llms.llm import LLM
from promptmodel.llms.llm_proxy import LLMProxy
from promptmodel.utils.types import LLMResponse, LLMStreamResponse

def test_stream(mocker):
	llm = LLM()
	test_messages = [
		{"role" : "system", "content" : "You are a helpful assistant."},
		{"role" : "user", "content" : "Introduce yourself in 50 words."},
	]
 
	stream_res : Generator[LLMStreamResponse, None, None] = llm.stream(
		messages=test_messages,
		functions=[],
		model="gpt-3.5-turbo",
	)
	error_count = 0
	api_responses : List[ModelResponse] = []
	for res in stream_res:
		if res.error:
			error_count += 1
			print("ERROR")
			print(res.error)
			print(res.error_log)
		if res.api_response:
			api_responses.append(res.api_response)

	assert error_count == 0, "error_count is not 0"
	assert len(api_responses) == 1, "api_count is not 1"
	
	assert api_responses[0].choices[0]['message']['content'] is not None, "content is None"
	assert api_responses[0]['response_ms'] is not None, "response_ms is None"
 
	# test logging
	llm_proxy = LLMProxy("test")
	
	mock_execute = AsyncMock()
	mock_response = MagicMock()
	mock_response.status_code = 200
	mock_execute.return_value = mock_response
	mocker.patch("promptmodel.llms.llm_proxy.AsyncAPIClient.execute", new=mock_execute)
	mocker.patch("promptmodel.llms.llm_proxy.read_config", new_callable=MagicMock, return_value={})
	
	llm_proxy._log_to_cloud(
		version_uuid="test",
		inputs={},
		api_response=api_responses[0],
		parsed_outputs={},
		metadata={},
	)
 
	mock_execute.assert_called_once()
	_, kwargs = mock_execute.call_args
 
	assert kwargs['json']['api_response'] == api_responses[0].to_dict_recursive(), "api_response is not equal"