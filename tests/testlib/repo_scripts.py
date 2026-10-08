"""
Import helpers for the repository's standalone scripts (util/, eval/, examples/), which are not packages.

    build_sam = load_repo_script("util/build_sam.py")
"""

import functools
import importlib.util
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@functools.cache
def load_repo_script(relpath: str):
    """Import a script by its path relative to the repository root, once per process. The module is registered
    in sys.modules under its basename so dataclasses and pickling inside it work"""
    path = os.path.join(REPO_DIR, relpath)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module
