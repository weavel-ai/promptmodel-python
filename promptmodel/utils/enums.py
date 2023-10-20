from enum import Enum

class LocalTask(str, Enum):
    RUN_LLM_MODULE = "RUN_LLM_MODULE"
    EVAL_LLM_MODULE = "EVAL_LLM_MODULE"
    LIST_MODULES = "LIST_MODULES"
    LIST_VERSIONS = "LIST_VERSIONS"
    LIST_SAMPLES = "LIST_SAMPLES"
    GET_PROMPTS = "GET_PROMPTS"
    GET_RUN_LOGS = "GET_RUN_LOGS"
    CHANGE_VERSION_STATUS = "CHANGE_VERSION_STATUS"
    GET_VERSION_TO_SAVE = "GET_VERSION_TO_SAVE"
    GET_VERSIONS_TO_SAVE = "GET_VERSIONS_TO_SAVE"
    UPDATE_CANDIDATE_VERSION_ID = "UPDATE_CANDIDATE_VERSION_ID"

class ServerTask(str, Enum):
    UPDATE_RESULT_RUN = "UPDATE_RESULT_RUN"
    LOCAL_UPDATE_ALERT = "LOCAL_UPDATE_ALERT"
    UPDATE_RESULT_EVAL = "UPDATE_RESULT_EVAL"
    
class LLMModuleVersionStatus(Enum):
    BROKEN = "broken"
    WORKING = "working"
    CANDIDATE = "candidate"
    
class ChangeLogAction(str, Enum):
    ADD: str = "ADD"
    DELETE: str = "DEL"
    CHANGE: str = "CHG"
    FIX: str = "FIX"

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class ParsingType(str, Enum):
    COLON = "colon" 
    SQUARE_BRACKET = "square_bracket"
    DOUBLE_SQUARE_BRACKET = "double_square_bracket"
    HTML = "html"
    
class ParsingPattern(dict, Enum):
    COLON = {
        "start" : r"({key}): \n",
        "end": None,
        "whole": r"({key}): \n(.*?)",
        "end_flag" : None,
        "refresh_flag" : None
    }
    SQUARE_BRACKET = {
        "start" : r"\[({key})\]",
        "end": r"[/{key}]",
        "whole": r"\[({key})\](.*?)\[/\1\]",
        "end_flag" : r"[",
        "refresh_flag" : r"]"
    }
    DOUBLE_SQUARE_BRACKET = {
        "start" : r"\[\[({key})\]\]",
        "end" : r"\[\[/{key}\]\]",
        "whole" : r"\[\[({key})\]\](.*?)\[\[/\1\]\]",
        "end_flag" : r"[",
        "refresh_flag" : r"]"
    }
    HTML = {
        "start" : r"<({key})>",
        "end" : r"</{key}>",
        "whole" : r"<({key})>(.*?)</\1>",
        "end_flag" : r"<",
        "refresh_flag" : r">"
    }

def get_pattern_by_type(parsing_type_value):
    return ParsingPattern[ParsingType(parsing_type_value).name].value
