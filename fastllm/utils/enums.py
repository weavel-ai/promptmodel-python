from enum import Enum

class LocalTask(str, Enum):
    RUN_LLM_MODULE = "RUN_LLM_MODULE"
    EVAL_LLM_MODULE = "EVAL_LLM_MODULE"
    LIST_MODULES = "LIST_MODULES"
    LIST_VERSIONS = "LIST_VERSIONS"
    GET_PROMPTS = "GET_PROMPTS"
    GET_RUN_LOGS = "GET_RUN_LOGS"

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
    