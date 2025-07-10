import json
from pathlib import Path

def save_session(
    filename: str,
    system_prompt: str,
    banned_strings: list[str],
    context: list[tuple[str, str | None]]
):
    """
    Save a single string and a list of strings to the given filename.
    Ensures the directory exists before writing. Expands ~ to your home dir.
    """
    path = Path(filename).expanduser()
    if path.parent:
        path.parent.mkdir(parents = True, exist_ok = True)

    payload = {
        "system_prompt": system_prompt,
        "banned_strings": banned_strings,
        "context": context
    }
    with path.open("w", encoding = "utf-8") as f:
        json.dump(payload, f, ensure_ascii = False, indent=2)


def load_session(filename: str):
    """
    Load and return (text, strings) from the given filename.
    Expands ~ to your home dir.
    """
    path = Path(filename).expanduser()
    with path.open("r", encoding = "utf-8") as f:
        data = json.load(f)
    return (
        data["system_prompt"],
        data["banned_strings"],
        data["context"]
    )