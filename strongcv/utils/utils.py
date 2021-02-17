import os
import json
from typing import Optional, Tuple, Union


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_json(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj: Union[list, dict], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, cls=NumpyArrayEncoder)


def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    columns, lines = default
    for fd in range(0, 3):  # First in order 0=Std In, 1=Std Out, 2=Std Error
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines
