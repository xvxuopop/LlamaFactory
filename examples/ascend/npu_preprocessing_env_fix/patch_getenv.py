import logging
import os


_seen = set()

_orig_getenv = os.getenv
_orig_environ_get = os.environ.get
loggerEnv = logging.getLogger("torch_npu.env")


def _log_once(key: str, val):
    if key in _seen:
        return
    _seen.add(key)
    loggerEnv.info(f"get env {key} = {val}")


def _patched_getenv(key, default=None):
    hit = key in os.environ
    val = _orig_getenv(key, default)
    if hit and isinstance(val, str) and val != "":
        _log_once(key, val)
    return val


def _patched_environ_get(key, default=None):
    hit = key in os.environ
    # Dataset.map(num_proc > 1) can replace os.environ with a deep copy.
    # Read the current mapping instead of a method bound to the old one.
    val = type(os.environ).get(os.environ, key, default)
    if hit and isinstance(val, str) and val != "":
        _log_once(key, val)
    return val


# patch on import
os.getenv = _patched_getenv
os.environ.get = _patched_environ_get
