import re
import sys
import pyperclip

def copy_last_codeblock(text: str) -> str | None:
    pattern = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    snippet = matches[-1].strip()
    pyperclip.copy(snippet)
    return snippet