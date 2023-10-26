from typing import Dict

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