import json
from typing import Any, Dict


def update_dict(
    target: Dict[str, str],
    source: Dict[str, str],
):
    for key, value in source.items():
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
        # Extract the inner type for the list
        inner_type = type_str[
            5:-1
        ]  # For "List[inner_type]", this extracts "inner_type"
        return [convert_str_to_type(item, inner_type) for item in json.loads(value)]
    elif type_str.startswith("Dict"):
        # Extract inner types for dict
        types = type_str[5:-1].split(
            ", "
        )  # For "Dict[key_type, value_type]", this extracts both
        loaded_dict: Dict[Any, Any] = json.loads(value)
        return {
            convert_str_to_type(k, types[0]): convert_str_to_type(v, types[1])
            for k, v in loaded_dict.items()
        }
    # Add other types as needed
    return value  # Default: Return as is
