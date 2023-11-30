import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from promptmodel.chat_model import RegisteringMeta


def test_registering_meta(mocker):
    # Fail to find DevClient instance
    client_instance = RegisteringMeta.find_client_instance()
    assert client_instance is None
