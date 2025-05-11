import sys, shutil

from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown
from rich.console import Console
from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.formatted_text import ANSI

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_think1 = "\u001b[35;1m"  # Bright magenta
col_think2 = "\u001b[35m"  # Magenta
col_error = "\u001b[31;1m"  # Bright red
col_info = "\u001b[32;1m"  # Bright red
col_sysprompt = "\u001b[37;1m"  # Grey

def print_error(text):
    print(col_error + "\nError: " + col_default + text)

def print_info(text):
    print(col_info + "\nInfo: " + col_default + text)

def read_input_console(args, user_name):
    print("\n" + col_user + user_name + ": " + col_default, end = '', flush = True)
    if args.multiline:
        user_prompt = sys.stdin.read().rstrip()
    else:
        user_prompt = input().strip()
    return user_prompt

def read_input_rich(args, user_name):
    user_prompt = Prompt.ask("\n" + col_user + user_name + col_default)
    return user_prompt

def read_input_ptk(args, user_name):
    print()
    user_prompt = ptk_prompt(ANSI(col_user + user_name + col_default + ": "), multiline = args.multiline)
    return user_prompt

class Streamer_basic:

    def __init__(self, args, bot_name):
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

    def stream(self, text: str, think_tag, end_think_tag):
        if self.all_text or not text.startswith(" "):
            print_text = text
        else:
            print_text = text[1:]
        self.all_text += text
        print(print_text, end = "", flush = True)

class MarkdownConsoleStream:

    def __init__(self, console: Console = None):
        # Make the Rich console a little narrower to prevent overflows from extra-wide emojis
        c, r = shutil.get_terminal_size(fallback = (80, 24))
        c -= 2
        self.console = console or Console(emoji_variant = "text", width = c)
        self.height = r - 2
        self._last_lines = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, markdown_text) -> None:
        new_lines = self._render_to_lines(markdown_text)
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
                # print()
        self._last_lines = new_lines

    def _render_to_lines(self, markdown_text: str):
        # Capture Rich’s output to a string, then split by lines.
        with self.console.capture() as cap:
            self.console.print(Markdown(markdown_text))
        rendered = cap.get()
        split = []
        for s in [r.rstrip() for r in rendered.rstrip("\n").split("\n")]:
            if s or len(split) == 0 or split[-1]:
                split.append(s)
        return split

    @staticmethod
    def _common_prefix_length(a, b) -> int:
        i = 0
        for x, y in zip(a, b):
            if x != y:
                break
            i += 1
        return i

class Streamer_rich:
    def __init__(self, args, bot_name):
        self.all_text = ""
        self.think_text = ""
        self.bot_name = bot_name
        self.all_print_text = col_bot + self.bot_name + col_default + ": "
        self.args = args
        self.live = None
        self.is_live = False

    def begin(self):
        self.live = MarkdownConsoleStream()
        self.live.__enter__()
        self.live.update(self.all_print_text)
        self.is_live = True

    def __enter__(self):
        if self.args.think:
            print()
            print(col_think1 + "Thinking" + col_default + ": " + col_think2, end = "")
        else:
            print()
            self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_live:
            self.live.__exit__(exc_type, exc_value, traceback)

    def stream(self, text: str, think_tag: str, end_think_tag: str):
        if self.args.think and not self.is_live:
            print_text = text
            if not self.think_text:
                print_text = print_text.lstrip()
            self.think_text += print_text
            if end_think_tag in self.think_text:
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
            formatted_text = formatted_text.replace(think_tag, f"`{think_tag}`")
            formatted_text = formatted_text.replace(end_think_tag, f"`{end_think_tag}`")
            self.live.update(formatted_text)