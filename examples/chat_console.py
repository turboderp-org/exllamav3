import sys, shutil
from pathlib import Path

from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.console import Console
from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
import time, os, re
import unicodedata

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_think1 = "\u001b[35;1m"  # Bright magenta
col_think2 = "\u001b[35m"  # Magenta
col_error = "\u001b[31;1m"  # Bright red
col_info = "\u001b[32;1m"  # Green
col_sysprompt = "\u001b[37;1m"  # Grey
col_dark = "\u001b[0;90m"  # White
col_b = "\u001b[100m"
LRO = '\u202D'  # Left-to-Right Override
PDF = '\u202C'  # Pop Directional Formatting

def get_history_file() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    path = base / "exllamav3"
    path.mkdir(parents = True, exist_ok = True)
    return path / "chat_history"

chat_history = FileHistory(str(get_history_file()))

def print_error(text):
    ftext = text.replace("\n", "\n       ")
    print(col_error + "\nError: " + col_default + ftext)

def print_info(text):
    ftext = text.replace("\n", "\n      ")
    print(col_info + "\nInfo: " + col_default + ftext)

def read_input_console(args, user_name, multiline: bool):
    print("\n" + col_user + user_name + ": " + col_default, end = '', flush = True)
    if multiline:
        user_prompt = sys.stdin.read().rstrip()
    else:
        user_prompt = input().strip()
    return user_prompt

def read_input_rich(args, user_name, multiline: bool):
    user_prompt = Prompt.ask("\n" + col_user + user_name + col_default)
    return user_prompt

def read_input_ptk(args, user_name, multiline: bool, prefix: str = None):
    print()
    user_prompt = ptk_prompt(
        ANSI(col_user + user_name + col_default + ": "),
        multiline = multiline,
        default = prefix or "",
        history = chat_history if not multiline else None,
    )
    return user_prompt

class Streamer_basic:

    def __init__(self, args, bot_name, think_tag, end_think_tag, updates_per_second, think):
        self.all_text = ""
        self.args = args
        self.bot_name = bot_name

    def __enter__(self):
        print()
        print(col_bot + self.bot_name + ": " + col_default, end = "")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.all_text.endswith("\n"):
            print()

    def stream(self, text: str):
        if self.all_text or not text.startswith(" "):
            print_text = text
        else:
            print_text = text[1:]
        self.all_text += text
        print(print_text, end = "", flush = True)

class MarkdownConsoleStream:
    """
    Incrementally rendered Markdown stream.

    Only the uncommitted tail is re-rendered per update, so the cost is bounded by one block or
    one code chunk regardless of response length. Output is line-diffed against the previous
    update and only the changed suffix is reprinted, as before.
    """

    CODE_CHUNK = 16

    _FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
    _LIST_RE = re.compile(r"^ {0,3}([-*+]|\d{1,9}[.)])(\s|$)")

    def __init__(self, console: Console = None):
        # Make the Rich console a little narrower to prevent overflows from extra-wide emojis
        c, r = shutil.get_terminal_size(fallback = (80, 24))
        c -= 2
        self.console = console or Console(emoji_variant = "text", width = c)
        self.height = r - 2
        self._last_lines = []
        self._reset()

    def _reset(self):
        self.src = ""                 # committed source (a prefix of the streamed text)
        self.lines = []               # rendered lines of self.src, no leading/trailing blanks
        self.fence = None             # open fence state when self.src ends inside one

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def _render_to_lines(self, markdown_text: str):
        # Capture Rich's output to a string, then split by lines (consecutive blanks collapsed)
        with self.console.capture() as cap:
            self.console.print(Markdown(markdown_text))
        rendered = cap.get()
        split = []
        for s in [r.rstrip() for r in rendered.rstrip("\n").split("\n")]:
            if s or len(split) == 0 or split[-1]:
                split.append(s)
        return split

    @staticmethod
    def _strip_blank(lines):
        a, b = 0, len(lines)
        while a < b and not lines[a]: a += 1
        while b > a and not lines[b - 1]: b -= 1
        return lines[a:b]

    def _render_block(self, text):
        return self._strip_blank(self._render_to_lines(text)) if text.strip() else []

    @staticmethod
    def _join(a, b, seam):
        if not a: return list(b)
        if not b: return list(a)
        return a + ([""] if seam else []) + b

    # ---- fence handling

    def _fence_open(self, line):
        m = self._FENCE_RE.match(line)
        if not m:
            return None
        return {"char": m.group(1)[0], "len": len(m.group(1)), "opener": line,
                "lang": m.group(2).strip().split(" ")[0] if m.group(2).strip() else ""}

    @staticmethod
    def _fence_lexer(fence):
        if "lexer" not in fence:
            from rich.syntax import Syntax
            try:
                fence["lexer"] = Syntax("", fence["lang"] or "text").lexer
            except Exception:
                fence["lexer"] = None
        return fence["lexer"]

    def _safe_lookback(self, fence, window, boundary):
        from pygments.token import String, Comment
        lexer = self._fence_lexer(fence)
        if lexer is None or boundary <= 0:
            return []
        text = "\n".join(window)
        off = len("\n".join(window[:boundary])) + 1
        toks = []
        for index, ttype, value in lexer.get_tokens_unprocessed(text):
            toks.append((index, ttype, value))
            if index >= off:
                break
        if not toks:
            return []
        def cat(t):
            return "s" if t in String else "c" if t in Comment else None
        start = None
        # A single token spanning the boundary (docstrings, block comments)
        for i, (index, ttype, value) in enumerate(toks):
            if index < off < index + len(value):
                start = i
                break
        if start is None:
            # Multi-line strings/comments some lexers emit as a run of same-category tokens
            # split at newlines: both sides of the boundary in the run -> start of the run
            before = next((i for i in range(len(toks) - 1, -1, -1) if toks[i][0] < off), None)
            after = next((i for i in range(len(toks)) if toks[i][0] >= off), None)
            if before is not None and after is not None and cat(toks[before][1]) is not None \
                    and cat(toks[before][1]) == cat(toks[after][1]):
                start = before
                while start > 0 and cat(toks[start - 1][1]) == cat(toks[before][1]):
                    start -= 1
        if start is None:
            return []
        start_line = text.count("\n", 0, toks[start][0])
        return window[start_line : boundary]

    def _fence_closes(self, fence, line):
        m = self._FENCE_RE.match(line)
        return bool(m) and m.group(1)[0] == fence["char"] and len(m.group(1)) >= fence["len"] \
            and not m.group(2).strip()

    def _code_render(self, fence, code_lines, closed):
        lb = fence["lookback"]
        closer = fence["char"] * fence["len"]
        body = "\n".join(lb + code_lines)
        text = fence["opener"] + "\n" + body + ("\n" + closer + "\n" if closed else "\n")
        out = self._render_block(text)
        if lb:
            # cached: the lookback rendered alone, minus its bottom padding line
            key = tuple(lb)
            if fence.get("lb_key") != key:
                fence["lb_key"] = key
                fence["lb_skip"] = len(self._render_block(fence["opener"] + "\n" + "\n".join(lb) + "\n")) - 1
            out = out[fence["lb_skip"]:]
        elif fence["committed"]:
            out = out[1:]                          # top padding belongs to the first chunk
        return out

    def _commit(self, tail):
        # Move as much of `tail` as possible into the committed state; return the remainder
        while True:
            lines = tail.split("\n")
            if self.fence is None:
                # Scan for the last safe blank-line split and for an open fence
                in_fence = None
                split_at = None
                fence_at = None
                for i, ln in enumerate(lines[:-1]):
                    f = self._fence_open(ln)
                    if in_fence is None and f:
                        in_fence, fence_at = f, i
                        continue
                    if in_fence is not None and self._fence_closes(in_fence, ln):
                        in_fence, fence_at = None, None
                        continue
                    if in_fence is None and ln.strip() == "":
                        nxt = lines[i + 1]
                        if nxt.startswith(("    ", "\t")) or self._LIST_RE.match(nxt):
                            continue
                        split_at = i
                if split_at is not None and (fence_at is None or split_at < fence_at):
                    head = "\n".join(lines[: split_at + 1]) + "\n"
                    self.lines = self._join(self.lines, self._render_block(head), True)
                    self.src += head
                    tail = tail[len(head):]
                    continue
                if fence_at is not None and len(lines) - 1 - (fence_at + 1) >= self.CODE_CHUNK:
                    # Open fence with a full chunk of completed lines: commit what precedes the
                    # opener as prose (a fence ends any paragraph), then open fence state
                    pre = "\n".join(lines[:fence_at]) + ("\n" if fence_at else "")
                    if pre:
                        self.lines = self._join(self.lines, self._render_block(pre), True)
                    f = self._fence_open(lines[fence_at])
                    f.update(lookback = [], committed = 0)
                    self.fence = f
                    self.src += pre + lines[fence_at] + "\n"
                    tail = tail[len(pre) + len(lines[fence_at]) + 1:]
                    continue
                return tail
            else:
                # Inside open fence: commit through a closer, else full chunks of lines
                fence = self.fence
                close_at = None
                for i, ln in enumerate(lines[:-1]):
                    if self._fence_closes(fence, ln):
                        close_at = i
                        break
                if close_at is not None:
                    code = lines[:close_at]
                    out = self._code_render(fence, code, True)
                    self.lines = self._join(self.lines, out, fence["committed"] == 0)
                    consumed = "\n".join(lines[: close_at + 1]) + "\n"
                    self.src += consumed
                    tail = tail[len(consumed):]
                    self.fence = None
                    continue
                done = len(lines) - 1
                if done >= self.CODE_CHUNK:
                    code = lines[: self.CODE_CHUNK]
                    out = self._code_render(fence, code, False)[:-1]      # drop bottom padding
                    # the block seam (blank line) belongs to the first chunk
                    self.lines = self._join(self.lines, out, fence["committed"] == 0)
                    consumed = "\n".join(code) + "\n"
                    self.src += consumed
                    tail = tail[len(consumed):]
                    fence["committed"] += len(code)
                    # Highlighter state for the next chunk: the window starts at a token
                    # boundary (the previous lookback start) and includes the unconsumed tail
                    # so a token continuing past the new boundary is visible
                    window = fence["lookback"] + code + lines[self.CODE_CHUNK:]
                    fence["lookback"] = self._safe_lookback(fence, window, len(fence["lookback"]) + len(code))
                    continue
                return tail

    @staticmethod
    def _common_prefix_length(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def update(self, markdown_text) -> None:
        if not markdown_text.startswith(self.src):
            self._reset()
        tail = self._commit(markdown_text[len(self.src):])
        if self.fence is None:
            new_lines = self._join(self.lines, self._render_block(tail), True)
        else:
            code = tail.split("\n")
            new_lines = self._join(self.lines, self._code_render(self.fence, code, False),
                                   self.fence["committed"] == 0)

        old_lines = self._last_lines
        prefix_length = self._common_prefix_length(old_lines, new_lines)
        prefix_length = max(prefix_length, len(old_lines) - self.height, len(new_lines) - self.height)
        old_suffix_len = len(old_lines) - prefix_length
        new_suffix_len = len(new_lines) - prefix_length
        if old_suffix_len > 0:
            print(f"{ESC}[{old_suffix_len}A", end = "")
        changed_count = max(old_suffix_len, new_suffix_len)
        for i in range(changed_count):
            if i < new_suffix_len:
                print(f"{ESC}[2K", end = "")  # Clear entire line
                print(new_lines[prefix_length + i].rstrip())
            else:
                print(f"{ESC}[2K", end = "")
        self._last_lines = new_lines

class Streamer_rich:
    def __init__(self, args, bot_name, think_tag, end_think_tag, updates_per_second, think):
        self.all_text = ""
        self.think_text = ""
        self.bot_name = bot_name
        self.all_print_text = col_bot + self.bot_name + col_default + ": "
        self.args = args
        self.live = None
        self.is_live = False
        self.think_tag = think_tag
        self.end_think_tag = end_think_tag
        self.updates_per_second = updates_per_second
        self.last_update = time.time()
        self.think = think

    def begin(self):
        self.live = MarkdownConsoleStream()
        self.live.__enter__()
        self.live.update(self.all_print_text)
        self.is_live = True

    def __enter__(self):
        if self.think and self.think_tag is not None:
            print()
            print(col_think1 + "Thinking" + col_default + ": " + col_think2, end = "")
        else:
            print()
            self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream("", True)
        if self.is_live:
            self.live.__exit__(exc_type, exc_value, traceback)

    def stream(self, text: str, force: bool = False):
        if self.think and self.think_tag is not None and not self.is_live:
            print_text = text
            if not self.think_text:
                print_text = print_text.lstrip()
            self.think_text += print_text
            if self.end_think_tag is not None and self.end_think_tag in self.think_text:
                print(print_text.rstrip(), flush = True)
                print()
                self.begin()
            else:
                print(print_text, end = "", flush = True)

        else:
            print_text = text
            if not self.all_text.strip():
                print_text = print_text.lstrip()
                if print_text.startswith("```"):
                    print_text = "\n" + print_text
            self.all_text += text
            self.all_print_text += print_text
            formatted_text = self.all_print_text
            if self.think_tag is not None:
                formatted_text = formatted_text.replace(self.think_tag, f"`{self.think_tag}`")
                formatted_text = formatted_text.replace(self.end_think_tag, f"`{self.end_think_tag}`")

            now = time.time()
            if now - self.last_update > 1.0 / self.updates_per_second or force:
                self.last_update = now
                self.live.update(formatted_text)


class Streamer_harmony:
    """Render each Harmony output channel as a separate Markdown block. Defaults to the GPT-OSS
    tags; other harmony-like formats override them via the `tags` dict (PromptFormat.harmony_tags):

        channel_tag:    opens a channel header
        message_tag:    ends the header, starts the message content
        end_tag:        ends a message within the turn (the final stop token is never streamed)
        prime:          text virtually prepended to the parser state, for formats whose first
                        channel header arrives without its opening tag (not added to all_text)
        channel_prefix: stripped from the displayed channel name
    """

    channel_tag = "<|channel|>"
    message_tag = "<|message|>"
    end_tag = "<|end|>"

    def __init__(self, args, bot_name, think_tag, end_think_tag, updates_per_second, think, tags = None):
        self.all_text = ""
        self.updates_per_second = updates_per_second
        self.last_update = time.time()
        self.buffer = ""
        self.channel = None
        self.channel_text = ""
        self.live = None
        self.channel_prefix = ""
        if tags:
            self.channel_tag = tags["channel_tag"]
            self.message_tag = tags["message_tag"]
            self.end_tag = tags["end_tag"]
            self.channel_prefix = tags.get("channel_prefix", "")
            self.buffer = tags.get("prime", "")

    def __enter__(self):
        print()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream("", True)
        self._close_channel(exc_type, exc_value, traceback)

    def _close_channel(self, exc_type = None, exc_value = None, traceback = None):
        if self.live is not None:
            self.live.update(self.channel_text)
            self.live.__exit__(exc_type, exc_value, traceback)
            self.live = None
            print()
        self.channel = None
        self.channel_text = ""

    def _begin_channel(self, channel):
        channel = channel.strip()
        if self.channel_prefix and channel.startswith(self.channel_prefix):
            channel = channel[len(self.channel_prefix):]
        self.channel = channel
        print(col_bot + self.channel + col_default + ":")
        print()
        self.live = MarkdownConsoleStream()
        self.live.__enter__()

    def _update(self, force = False):
        if self.live is None:
            return
        now = time.time()
        if force or now - self.last_update > 1.0 / self.updates_per_second:
            self.last_update = now
            self.live.update(self.channel_text)

    def stream(self, text: str, force: bool = False):
        self.all_text += text
        self.buffer += text

        while True:
            if self.channel is None:
                pos = self.buffer.find(self.channel_tag)
                if pos < 0:
                    # Retain enough of the suffix to recognize a split tag.
                    self.buffer = self.buffer[-(len(self.channel_tag) - 1):]
                    break
                self.buffer = self.buffer[pos + len(self.channel_tag):]
                pos = self.buffer.find(self.message_tag)
                if pos < 0:
                    self.buffer = self.channel_tag + self.buffer
                    break
                self._begin_channel(self.buffer[:pos])
                self.buffer = self.buffer[pos + len(self.message_tag):]

            pos = self.buffer.find(self.end_tag)
            if pos < 0:
                keep = len(self.end_tag) - 1
                if len(self.buffer) > keep:
                    self.channel_text += self.buffer[:-keep]
                    self.buffer = self.buffer[-keep:]
                break

            self.channel_text += self.buffer[:pos]
            self.buffer = self.buffer[pos + len(self.end_tag):]
            self._close_channel()

        if force and self.channel is not None:
            self.channel_text += self.buffer
            self.buffer = ""
        self._update(force)


class KeyReader:
    """
    Cross-platform, non-blocking key reader.
    Usage:
        with KeyReader() as keys:
            k = keys.getkey()        # non-blocking (returns None if no key)
            k = keys.getkey(timeout) # wait up to `timeout` seconds
    """
    def __init__(self):
        self._platform = 'nt' if os.name == 'nt' else 'posix'
        self._entered = False
        # POSIX fields
        self._old_termios = None
        self.disabled = False

    def __enter__(self):
        try:
            if self._platform == 'posix':
                import termios, tty
                self._termios = termios
                self._tty = tty
                self._fd = sys.stdin.fileno()
                self._old_termios = self._termios.tcgetattr(self._fd)
                # cbreak mode lets us read single characters without Enter
                self._tty.setcbreak(self._fd)
            self._entered = True
            return self
        except termios.error:
            self.disabled = True
            return self

    def __exit__(self, exc_type, exc, tb):
        if not self.disabled and self._platform == 'posix' and self._old_termios:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_termios)
        self._entered = False

    def getkey(self, timeout=0.0):
        """
        Returns a single-character string if a key is pressed, or None.
        `timeout` is seconds to wait (float). 0.0 => non-blocking poll.
        """
        if self.disabled:
            return None

        if not self._entered:
            raise RuntimeError("Use KeyReader as a context manager")

        if self._platform == 'nt':
            import msvcrt
            # busy-wait with tiny sleeps if a timeout is requested
            end = None if timeout is None else (time.time() + timeout)
            while True:
                if msvcrt.kbhit():
                    b = msvcrt.getch()
                    # Handle special keys (arrows, function keys) which come as a prefix + code
                    if b in (b'\x00', b'\xe0'):  # special prefix
                        _ = msvcrt.getch()       # consume the second byte
                        return None              # ignore special keys for simplicity
                    try:
                        return b.decode('utf-8', errors='ignore').lower()
                    except Exception:
                        return None
                if timeout == 0.0:
                    return None
                if end is not None and time.time() >= end:
                    return None
                time.sleep(0.005)
        else:
            import select
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if r:
                try:
                    ch = sys.stdin.read(1)
                except (IOError, OSError):
                    return None
                return (ch or "").lower() or None
            return None


def display_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1 for c in s)

def char_width(c):
    return 2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1

def ljust_truncate(s, width):
    dw = display_width(s)
    if dw <= width:
        return s + ' ' * (width - dw)
    max_content = width - 1  # reserve 1 column for '…'
    cols = 0
    truncated = []
    for c in s:
        cw = char_width(c)
        if cols + cw > max_content:
            break
        truncated.append(c)
        cols += cw
    padding = width - cols - 1  # 1 for the ellipsis
    t = ''.join(truncated) + '…' + ' ' * padding
    return t

def print_tokens(
    ids: list,
    vocab: list,
    ids_per_line = 10,
):
    print()
    line = ""
    for pos in range(len(ids)):
        t = ids[pos]
        p = repr(vocab[t])[1:-1].replace(" ", "␣")
        line += f"{col_user}{t:6}{col_default} "
        line += f"{ljust_truncate(p, 10)} "
        if (pos + 1) % ids_per_line == 0 or pos == len(ids) - 1:
            line = line.replace("␣", f"{col_bot}␣{col_default}").replace("…", f"{col_error}…{col_default}")
            ppos = pos // ids_per_line * ids_per_line
            print(f"{col_info}{ppos:6} {col_default}: {line}")
            line = ""

def print_probs(
    saved_topk: list,
    saved_probs: list,
    saved_samples: list,
    vocab: list,
    ids_per_line = 5,
    minimum_p = 1e-6,
):
    if len(saved_topk) == 0:
        return
    num = len(saved_topk[0])
    lines = ["    "] * (num + 2)
    ids_this_line = 0
    for ids, probs, sample in zip(saved_topk, saved_probs, saved_samples):
        tss = repr(vocab[sample[0]])[1:-1].replace(" ", "␣")
        lines[0] += f"{col_sysprompt}" + ljust_truncate(tss, 27) + f"{col_default}  "
        lines[1] += f"{col_dark}" + "─" * 27 + f"{col_default}  "
        for i, (t, p) in enumerate(zip(ids, probs)):
            if p < minimum_p:
                lines[i + 2] += " " * 29
            else:
                hl = col_b if sample[0] == t else ""
                ts = repr(vocab[t])[1:-1].replace(" ", "␣")
                lines[i + 2] += f"{col_user}{hl}  {t:6}{col_default}{hl} "
                lines[i + 2] += f"{ljust_truncate(ts, 10)} "
                lines[i + 2] += f"{col_think1}{hl}{p:7.5f}{col_default}  "
        ids_this_line += 1
        if ids_this_line == ids_per_line:
            print(f"\n{LRO}" + "\n".join(lines) + f"{PDF}")
            lines = ["    "] * (num + 2)
            ids_this_line = 0
    if ids_this_line > 0:
        print(f"\n{LRO}" + "\n".join(lines) + f"{PDF}")
