import os
import shelve
import sys
import json
from typing import Any, Dict, List, TypeVar, Union, cast
from functools import lru_cache, wraps


def disk_lru_cache_name(filename):
    directory = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "__disk_lru_cache__")
    os.makedirs(directory, exist_ok = True)
    path = os.path.join(directory, filename + ".lru")
    return str(path)

def disk_lru_cache_clear(filename, *args, **kwargs):
    """
    Clear disk cache. Takes name of function and input arguments to forget.
    """
    with shelve.open(disk_lru_cache_name(filename)) as db:
        key = str((json.dumps(args, sort_keys = True), json.dumps(kwargs, sort_keys = True)))
        if key in db:
            del db[key]

def disk_lru_cache(filename):
    """
    Disk cache function decorator, mostly for quick-and-dirty caching of eval results. Creates a
    __disk_lru_cache__  subdirectory in the module's directory to store cached outputs. The cache dictionary
    is identified by the name of the calling function, so multiple functions in the same directory with the
    same name would result in a conflict.

    Requires all arguments to have a full string representation, and the function output must be
    serializable by the shelve library.
    """
    def decorator(func):
        @wraps(func)
        def disk_cached(*args, **kwargs):
            with shelve.open(disk_lru_cache_name(filename)) as db:
                key = str((json.dumps(args, sort_keys = True), json.dumps(kwargs, sort_keys = True)))
                if key in db:
                    return db[key]
                result = func(*args, **kwargs)
                db[key] = result
                return result
        return disk_cached
    return decorator

no_default = object()
no_value = object()
T = TypeVar('T')


def read_dict(
        input_dict: dict[str, Any],
        expected_types: type | list[type],
        keys: str | list[str],
        default = no_default,
) -> T:
    """
    Utility function to read typed value from (nested) dictionary

    :param input_dict:
        The dict to read from

    :param expected_types:
        Type or list of types to expect the value to be. Raise an exception if the key exists but value is
        of the wrong type. If expected_type is None, any value type is accepted

    :param keys:
        Key or list of keys to look for. If multiple keys in the list would match, the first matching key is
        used. Keys can index nested dictionaries with a "->" separator, e.g. "text_model->hidden_size" would
        be equivalent to dict["text_model"]["hidden_size"]

    :param default:
        Default value to return if the key isn't found, e.g. None. If this is the special value no_default
        and no keys are matched, raise an exception instead.

    :return:
        Requested value if key found, otherwise default value
    """

    if expected_types is not None and not isinstance(expected_types, list):
        expected_types = [expected_types]

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        input_dict_s = input_dict

        key_split = key.split("->")
        for subk in key_split[:-1]:
            input_dict_s = input_dict_s.get(subk, None)
            if not input_dict_s:
                key = None
                break
        if key is None: continue
        key = key_split[-1]

        x = input_dict_s.get(key, None)
        if x is not None:
            if expected_types is None:
                return x
            else:
                for t in expected_types:
                    # Always cast int to float
                    if t == float and isinstance(x, int):
                        x = float(x)
                    # Allow casting float to int if no rounding error
                    if t == int and isinstance(x, float) and x == int(x):
                        x = int(x)
                    if isinstance(x, t):
                        return cast(T, x)
            raise TypeError(f"Value for {key} is not of expected type: {expected_types}")

    if default != no_default:
        return default
    raise ValueError(f"Missing any of the following keys: {keys}")


def maybe_read_json(path):
    """
    Read JSON file to dict, or return empty dict if file does not exist
    """
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
    else:
        return {}
