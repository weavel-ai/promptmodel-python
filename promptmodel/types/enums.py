from enum import Enum


class InstanceType(str, Enum):
    ChatLog = "ChatLog"
    RunLog = "RunLog"
    ChatLogSession = "ChatLogSession"


class LocalTask(str, Enum):
    RUN_PROMPT_MODEL = "RUN_PROMPT_MODEL"
    RUN_CHAT_MODEL = "RUN_CHAT_MODEL"

    LIST_CODE_CHAT_MODELS = "LIST_CHAT_MODELS"
    LIST_CODE_PROMPT_MODELS = "LIST_PROMPT_MODELS"
    LIST_CODE_FUNCTIONS = "LIST_FUNCTIONS"


class LocalTaskErrorType(str, Enum):
    NO_FUNCTION_NAMED_ERROR = "NO_FUNCTION_NAMED_ERROR"  # no DB update is needed
    FUNCTION_CALL_FAILED_ERROR = "FUNCTION_CALL_FAILED_ERROR"  # create FunctionModelVersion, create Prompt, create RunLog
    PARSING_FAILED_ERROR = "PARSING_FAILED_ERROR"  # create FunctionModelVersion, create Prompt, create RunLog

    SERVICE_ERROR = "SERVICE_ERROR"  # no DB update is needed


class ServerTask(str, Enum):
    UPDATE_RESULT_RUN = "UPDATE_RESULT_RUN"
    UPDATE_RESULT_CHAT_RUN = "UPDATE_RESULT_CHAT_RUN"

    SYNC_CODE = "SYNC_CODE"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ParsingType(str, Enum):
    COLON = "colon"
    SQUARE_BRACKET = "square_bracket"
    DOUBLE_SQUARE_BRACKET = "double_square_bracket"
    HTML = "html"


class ParsingPattern(dict, Enum):
    COLON = {
        "start": r"(\w+)\s+type=([\w,\s\[\]]+): \n",
        "start_fstring": "{key}: \n",
        "end_fstring": None,
        "whole": r"(.*?): (.*?)\n",
        "start_token": None,
        "end_token": None,
    }
    SQUARE_BRACKET = {
        "start": r"\[(\w+)\s+type=([\w,\s\[\]]+)\]",
        "start_fstring": "[{key} type={type}]",
        "end_fstring": "[/{key}]",
        "whole": r"\[(\w+)\s+type=([\w,\s\[\]]+)\](.*?)\[/\1\]",
        "start_token": r"[",
        "end_token": r"]",
    }
    DOUBLE_SQUARE_BRACKET = {
        "start": r"\[\[(\w+)\s+type=([\w,\s\[\]]+)\]\]",
        "start_fstring": "[[{key} type={type}]]",
        "end_fstring": "[[/{key}]]",
        "whole": r"\[\[(\w+)\s+type=([\w,\s\[\]]+)\]\](.*?)\[\[/\1\]\]",
        "start_token": r"[",
        "end_token": r"]",
    }
    HTML = {
        "start": r"<(\w+)\s+type=([\w,\s\[\]]+)>",
        "start_fstring": "<{key} type={type}>",
        "end_fstring": "</{key}>",
        "whole": r"<(\w+)\s+type=([\w,\s\[\]]+)>(.*?)</\1>",  # also captures type
        "start_token": r"<",
        "end_token": r">",
    }


def get_pattern_by_type(parsing_type_value):
    return ParsingPattern[ParsingType(parsing_type_value).name].value
