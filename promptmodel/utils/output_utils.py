import json
from typing import Any, Dict


def update_dict(
    target: Dict[str, str],
    source: Dict[str, str],
):
    for key, value in source.items():
        if value is not None:
            if key not in target:
                target[key] = value
            else:
                target[key] += value
    return target


def convert_str_to_type(value: str, type_str: str) -> Any:
    if type_str == "str":
        return value.strip()
    elif type_str == "bool":
        return value.lower() == "true"
    elif type_str == "int":
        return int(value)
    elif type_str == "float":
        return float(value)
    elif type_str.startswith("List"):
        return json.loads(value)
    elif type_str.startswith("Dict"):
        return json.loads(value)
    return value  # Default: Return as is
