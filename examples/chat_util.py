import re
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

def extract_svg(s: str, begin: str = "<svg", end: str = "</svg>"):

    # Find all tag occurrences in order
    pattern = re.compile(rf"{re.escape(begin)}|{re.escape(end)}")
    tags = list(pattern.finditer(s))

    best = None
    for i in range(len(tags) - 1):
        t1, t2 = tags[i], tags[i+1]
        if t1.group() == begin and t2.group() == end:
            start = t1.start()
            stop  = t2.end()
            length = stop - start
            if best is None or length > best[0]:
                best = (length, start, stop)

    if not best:
        return None

    _, start, stop = best
    return s[start:stop]