from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize = 8)
def _load_template(directory):
    try:
        from jinja2 import nodes
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        raise RuntimeError("Install exllamav3[examples] to render the Olala chat template") from exc

    class GenerationBlock(Extension):
        tags = {"generation"}

        def parse(self, parser):
            token = next(parser.stream)
            body = parser.parse_statements(("name:endgeneration",), drop_needle = True)
            return nodes.Scope(body).set_lineno(token.lineno)

    env = ImmutableSandboxedEnvironment(
        extensions = [GenerationBlock], trim_blocks = True, lstrip_blocks = True,
    )
    return env.from_string((Path(directory) / "chat_template.jinja").read_text(encoding = "utf-8"))


def render_olala_chat(directory, messages, reasoning_effort = "medium"):
    return _load_template(Path(directory).resolve()).render(
        messages = messages, add_generation_prompt = True, reasoning_effort = reasoning_effort,
        tools = None, preserve_thinking = False,
    )
