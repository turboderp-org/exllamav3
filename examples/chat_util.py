import re
import sys
import pyperclip

def copy_last_codeblock(text: str, num) -> str | None:
    pattern = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    if num > len(matches):
        num = len(matches)
    snippet = matches[-num].strip()
    pyperclip.copy(snippet)
    return snippet