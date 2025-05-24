"""
Turn @profile into a no-op when it's not injected by kernprof
"""

import builtins

if not hasattr(builtins, "profile"):
    def _noop(func):
        return func
    builtins.profile = _noop

profile = builtins.profile
__all__ = ["profile"]
