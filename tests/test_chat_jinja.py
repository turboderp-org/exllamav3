"""
CPU tests for training/chat_jinja.py -- the Jinja chat-template renderer
behind ``--prompt-format jinja``:

  * template resolution from a model directory (chat_template.jinja /
    chat_template.json / tokenizer_config.json, named-template lists,
    explicit override file);
  * incremental segment rendering against a Qwen-style ChatML template:
    concatenation must equal the full-conversation render, only assistant
    bodies (reasoning span, content, tool-call block, turn close) may be
    supervised, headers/prefills and inter-turn separators masked;
  * rich inputs: tool roles, tool_calls (incl. OAI wire-format JSON-string
    arguments), reasoning_content, top-level tools, template_vars /
    chat_template_kwargs (incl. enable_thinking header splitting);
  * prefix-monotonicity violations (history re-rendering) raising ValueError
    so callers skip-and-count instead of mislabeling;
  * multimodal content-parts passthrough (placeholders render, text parts
    feed turn_text word counts);
  * eot derivation and the jinja_renderers trainer facade.

Needs jinja2 (a torch dependency; pip install jinja2 if standalone). No
GPU / compiled extension / real model needed. Run:
    python tests/test_chat_jinja.py
"""

from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, "training", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cj = _load("chat_jinja")
ct = _load("chat_turns")

try:
    import jinja2  # noqa: F401
except ImportError:
    print("jinja2 not importable; skipping chat_jinja tests")
    sys.exit(0)


def u(c): return {"role": "user", "content": c}
def a(c): return {"role": "assistant", "content": c}
def s(c): return {"role": "system", "content": c}


def cat(segs): return "".join(t for t, _ in segs)
def sup(segs): return [t for t, is_sup in segs if is_sup]


# A Qwen-style ChatML template exercising every rich path: tools in the
# system block, reasoning_content, tool_calls, tool responses, an
# enable_thinking-gated generation prompt, and raise_exception.
CHATML_TPL = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\n' }}"
    "{%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}"
    "{{- '# Tools\n' }}"
    "{%- for tool in tools %}{{- tool | tojson }}{{- '\n' }}{%- endfor %}"
    "{{- '<|im_end|>\n' }}"
    "{%- elif messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' %}"
    "{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- if message.reasoning_content %}"
    "{{- '<think>\n' + message.reasoning_content + '\n</think>\n\n' }}"
    "{%- endif %}"
    "{{- message.content }}"
    "{%- for tc in message.tool_calls or [] %}"
    "{{- '\n<tool_call>\n{\"name\": \"' + tc.function.name + '\", \"arguments\": ' + (tc.function.arguments | tojson) + '}\n</tool_call>' }}"
    "{%- endfor %}"
    "{{- '<|im_end|>\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}"
    "{%- elif message.role != 'system' %}"
    "{{- raise_exception('unknown role: ' + message.role) }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and not enable_thinking %}"
    "{{- '<think>\n\n</think>\n\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)


def make_build(tpl=CHATML_TPL, default_vars=None):
    return cj.make_jinja_segment_builder(cj.compile_chat_template(tpl),
                                         default_vars)


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def test_load_chat_template():
    with tempfile.TemporaryDirectory() as d:
        def write(fn, content):
            with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                f.write(content if isinstance(content, str)
                        else json.dumps(content))
        # nothing anywhere -> ValueError with a pointer
        try:
            cj.load_chat_template(d)
        except ValueError as e:
            assert "chat-template-file" in str(e)
        else:
            raise AssertionError("expected ValueError")
        # tokenizer_config.json, plain string
        write("tokenizer_config.json", {"chat_template": "TC"})
        assert cj.load_chat_template(d) == "TC"
        # named-template list: default picked, names selectable, missing fails
        write("tokenizer_config.json", {"chat_template": [
            {"name": "default", "template": "DEF"},
            {"name": "tool_use", "template": "TOOLS"}]})
        assert cj.load_chat_template(d) == "DEF"
        assert cj.load_chat_template(d, name="tool_use") == "TOOLS"
        try:
            cj.load_chat_template(d, name="rag")
        except ValueError as e:
            assert "tool_use" in str(e)
        else:
            raise AssertionError("expected ValueError")
        # chat_template.json (multimodal processor convention) wins over
        # tokenizer_config.json
        write("chat_template.json", {"chat_template": "CTJ"})
        assert cj.load_chat_template(d) == "CTJ"
        # chat_template.jinja wins over both
        write("chat_template.jinja", "CTJINJA")
        assert cj.load_chat_template(d) == "CTJINJA"
        # explicit override file wins over everything
        write("custom.jinja", "CUSTOM")
        assert cj.load_chat_template(
            d, template_file=os.path.join(d, "custom.jinja")) == "CUSTOM"
    print("load_chat_template: OK")


# ---------------------------------------------------------------------------
# Segment rendering
# ---------------------------------------------------------------------------

def test_single_turn():
    build = make_build()
    render = cj.compile_chat_template(CHATML_TPL)
    turns = [s("S"), u("Q"), a("R")]
    segs = build(turns)
    # concatenation == the full-conversation render, bit for bit
    assert cat(segs) == render(turns), f"{cat(segs)!r}"
    assert cat(segs) == ("<|im_start|>system\nS<|im_end|>\n"
                         "<|im_start|>user\nQ<|im_end|>\n"
                         "<|im_start|>assistant\nR<|im_end|>\n")
    # ONLY the assistant body + close supervised; trailing \n masked
    assert sup(segs) == ["R<|im_end|>"], f"{segs!r}"
    print("single-turn: OK")


def test_multi_turn():
    build = make_build()
    render = cj.compile_chat_template(CHATML_TPL)
    turns = [u("Q1"), a("R1"), u("Q2"), a("R2")]
    segs = build(turns)
    assert cat(segs) == render(turns)
    assert sup(segs) == ["R1<|im_end|>", "R2<|im_end|>"]
    # the \n after an assistant close is carried into the NEXT masked segment
    # (the inter-turn separator convention the hardcoded formats use)
    assert ("\n<|im_start|>user\nQ2<|im_end|>\n", False) in segs
    assert segs[-1] == ("\n", False)
    print("multi-turn: OK")


def test_generation_prompt():
    build = make_build()
    segs = build([u("Q1"), a("R1"), u("Q2")], add_generation_prompt=True)
    assert cat(segs).endswith("<|im_start|>user\nQ2<|im_end|>\n"
                              "<|im_start|>assistant\n")
    assert not segs[-1][1]
    # enable_thinking=False puts the empty think prefill in the MASKED tail
    segs = build([u("Q")], add_generation_prompt=True,
                 template_vars={"enable_thinking": False})
    assert cat(segs).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert sup(segs) == []
    print("generation prompt: OK")


def test_nothink_header_split():
    # With enable_thinking=False the add_generation_prompt render prefills the
    # empty think block but the assistant-message render doesn't emit it: the
    # longest-common-prefix split must still put the header in the mask and
    # only the content + close under supervision.
    build = make_build()
    segs = build([u("Q"), a("R")], template_vars={"enable_thinking": False})
    assert ("<|im_start|>assistant\n", False) in segs
    assert sup(segs) == ["R<|im_end|>"]
    print("nothink header split: OK")


def test_reasoning_supervised():
    build = make_build()
    turns = [u("Q"), {"role": "assistant", "content": "R",
                      "reasoning_content": "RC"}]
    segs = build(turns)
    assert sup(segs) == ["<think>\nRC\n</think>\n\nR<|im_end|>"]
    print("reasoning_content supervised: OK")


def test_tool_flow():
    build = make_build()
    tc = {"type": "function",
          "function": {"name": "get_x", "arguments": {"q": 1}}}
    turns = [u("Q"),
             {"role": "assistant", "content": "", "tool_calls": [tc]},
             {"role": "tool", "content": "42"},
             a("the answer is 42")]
    tools = [{"type": "function",
              "function": {"name": "get_x", "parameters": {}}}]
    segs = build(turns, tools=tools)
    render = cj.compile_chat_template(CHATML_TPL)
    assert cat(segs) == render(turns, tools=tools)
    # tools land in the (masked) system block
    assert segs[0][1] is False and "# Tools" in segs[0][0]
    # the tool-call block + turn close is SUPERVISED (this is what makes the
    # model emit tool calls -- and the close token is what trips the
    # tool-call finish reason at inference); the tool response is masked
    assert sup(segs) == [
        '\n<tool_call>\n{"name": "get_x", "arguments": {"q": 1}}\n'
        '</tool_call><|im_end|>',
        "the answer is 42<|im_end|>"]
    assert ("\n<|im_start|>user\n<tool_response>\n42\n</tool_response>"
            "<|im_end|>\n", False) in segs
    print("tool flow: OK")


def test_template_vars_layering():
    # builder-level default_vars are overridden per call by template_vars
    build = make_build(default_vars={"enable_thinking": False})
    segs = build([u("Q")], add_generation_prompt=True)
    assert cat(segs).endswith("<think>\n\n</think>\n\n")
    segs = build([u("Q")], add_generation_prompt=True,
                 template_vars={"enable_thinking": True})
    assert cat(segs).endswith("<|im_start|>assistant\n")
    print("template_vars layering: OK")


def test_special_tokens_in_context():
    build = make_build(tpl="{{ bos_token }}{% for m in messages %}"
                           "[{{ m.role }}]{{ m.content }}{% endfor %}",
                       default_vars={"bos_token": "<s>"})
    segs = build([u("Q"), a("R")])
    assert cat(segs).startswith("<s>[user]Q")
    print("special tokens in context: OK")


def test_unrenderable():
    build = make_build()
    # raise_exception in the template -> ValueError (skip-and-count)
    for bad in ([u("q"), {"role": "critic", "content": "x"}, a("r")], []):
        try:
            build(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    # message shape the template can't handle (content=None) -> ValueError,
    # not a crash
    try:
        build([{"role": "user", "content": None}, a("r")])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    # empty assistant CONTENT still renders its turn close -> that close is
    # the supervised body (extract_rich_turns drops truly-empty turns before
    # build ever sees them)
    assert sup(build([u("q"), a("")])) == ["<|im_end|>"]
    # but an assistant turn the template renders NO text for at all raises
    bare = make_build(tpl="{%- for m in messages %}{{ m.content }}"
                          "{%- endfor %}")
    try:
        bare([u("q"), a("")])
    except ValueError as e:
        assert "empty" in str(e)
    else:
        raise AssertionError("expected ValueError")
    print("unrenderable shapes: OK")


def test_deferred_prefix():
    # Qwen3.6/3.8 templates raise_exception when the prefix has no user turn
    # yet, so messages[:1] (system only) can't render. The builder defers
    # that prefix into the next turn's masked segment; the result still
    # concatenates to the full render with only the assistant body
    # supervised. A conversation whose LAST turn can't render still raises.
    guard = ("{%- set ns = namespace(user=false) %}"
             "{%- for m in messages %}{%- if m.role == 'user' %}"
             "{%- set ns.user = true %}{%- endif %}{%- endfor %}"
             "{%- if not ns.user %}"
             "{{ raise_exception('No user query found in messages.') }}"
             "{%- endif %}")
    tpl = guard + CHATML_TPL
    build = make_build(tpl=tpl)
    render = cj.compile_chat_template(tpl)
    turns = [s("S"), u("Q"), a("R")]
    segs = build(turns)
    assert cat(segs) == render(turns), f"{cat(segs)!r}"
    assert sup(segs) == ["R<|im_end|>"], f"{segs!r}"
    assert segs[0] == ("<|im_start|>system\nS<|im_end|>\n"
                       "<|im_start|>user\nQ<|im_end|>\n", False), segs[0]
    try:
        build([s("S")])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("deferred prefix: OK")


def test_prefix_violation():
    # A template that keeps reasoning only on the LAST message re-renders
    # history: turns[:2] shows the think span, turns[:3] drops it. That must
    # raise (with a pointer), not silently mislabel.
    tpl = ("{%- for message in messages %}"
           "{%- if message.role == 'assistant' %}"
           "{{- '<A>' }}"
           "{%- if message.reasoning_content and loop.last %}"
           "{{- '<think>' + message.reasoning_content + '</think>' }}"
           "{%- endif %}"
           "{{- message.content + '</A>\n' }}"
           "{%- else %}"
           "{{- '<U>' + message.content + '</U>\n' }}"
           "{%- endif %}"
           "{%- endfor %}"
           "{%- if add_generation_prompt %}{{- '<A>' }}{%- endif %}")
    build = make_build(tpl=tpl)
    bad = [u("Q1"), {"role": "assistant", "content": "R1",
                     "reasoning_content": "RC"}, u("Q2"), a("R2")]
    try:
        build(bad)
    except ValueError as e:
        assert "reasoning_content" in str(e)
    else:
        raise AssertionError("expected ValueError")
    # reasoning only on the final assistant turn (the OAI-history shape)
    # renders fine under the same template
    ok = [u("Q1"), a("R1"), u("Q2"),
          {"role": "assistant", "content": "R2", "reasoning_content": "RC"}]
    segs = build(ok)
    # the <A> opener is the generation prompt -> masked header; reasoning +
    # content + close supervised
    assert sup(segs) == ["R1</A>", "<think>RC</think>R2</A>"]
    print("prefix violation: OK")


def test_multimodal_content_parts():
    tpl = ("{%- for message in messages %}"
           "{{- '<' + message.role + '>' }}"
           "{%- if message.content is string %}"
           "{{- message.content }}"
           "{%- else %}"
           "{%- for p in message.content %}"
           "{%- if p.type == 'image' %}{{- '<|image|>' }}"
           "{%- elif p.type == 'text' %}{{- p.text }}"
           "{%- endif %}"
           "{%- endfor %}"
           "{%- endif %}"
           "{{- '</' + message.role + '>\n' }}"
           "{%- endfor %}"
           "{%- if add_generation_prompt %}{{- '<assistant>' }}{%- endif %}")
    build = make_build(tpl=tpl)
    mm_user = {"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "what is this?"}]}
    turns = [mm_user, a("a cat")]
    segs = build(turns)
    # the image placeholder renders into the MASKED user turn
    assert ("<user><|image|>what is this?</user>\n", False) in segs
    assert sup(segs) == ["a cat</assistant>"]
    assert cj.has_nontext_parts(turns) is True
    assert cj.has_nontext_parts([u("q"), a("r")]) is False
    assert ct.turn_text(mm_user) == "what is this?"
    assert ct.turn_text(a("a cat")) == "a cat"
    print("multimodal content-parts: OK")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def test_extract_rich_turns():
    msgs = [
        {"role": "System", "content": " sys ", "tool_calls": None},
        {"role": "USER", "content": "hi", "reasoning_content": None},
        {"role": "assistant", "content": "", "tool_calls": []},  # empty -> dropped
        {"role": "assistant", "content": "yo ", "reasoning_content": " think "},
        # flat legacy tool call, arguments as OAI wire-format JSON string
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": None, "name": "f",
                         "arguments": "{\"x\": 2}"}]},
        {"role": "tool", "content": "42", "tool_call_id": "1"},
    ]
    turns = cj.extract_rich_turns(msgs)
    assert turns[0] == {"role": "system", "content": "sys"}
    assert turns[1] == {"role": "user", "content": "hi"}
    assert turns[2] == {"role": "assistant", "content": "yo",
                        "reasoning_content": "think"}
    assert turns[3] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "type": "function",
         "function": {"name": "f", "arguments": {"x": 2}}}]}
    assert turns[4] == {"role": "tool", "content": "42", "tool_call_id": "1"}
    assert cj.extract_rich_turns(None) == []
    # nested shape with dict arguments passes through untouched
    nested = {"role": "assistant", "content": "c", "tool_calls": [
        {"type": "function", "function": {"name": "g", "arguments": {"y": 1}}}]}
    assert cj.extract_rich_turns([nested])[0]["tool_calls"][0]["function"] == \
        {"name": "g", "arguments": {"y": 1}}
    print("extract_rich_turns: OK")


def test_row_template_extras():
    row = {
        "messages": [u("q")],
        "tools": '[{"type": "function", "function": {"name": "f"}}]',
        "chat_template_kwargs": {"a": 2, "b": 3, "add_generation_prompt": True},
        "template_vars": {"a": 1, "tools": ["shadowed"]},
    }
    x = cj.row_template_extras(row)
    # JSON-string tools parsed; top-level tools wins over a var-bag tools
    assert x["tools"] == [{"type": "function", "function": {"name": "f"}}]
    # template_vars wins over chat_template_kwargs per key; reserved keys gone
    assert x["template_vars"] == {"a": 1, "b": 3}
    # tools inside a var bag is honored when there's no top-level tools
    x = cj.row_template_extras({"template_vars": {"tools": ["T"], "k": 1}})
    assert x["tools"] == ["T"] and x["template_vars"] == {"k": 1}
    assert cj.row_template_extras({}) == {"tools": None, "template_vars": None}
    print("row_template_extras: OK")


# ---------------------------------------------------------------------------
# Trainer facade
# ---------------------------------------------------------------------------

def test_jinja_renderers():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "tokenizer_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"chat_template": CHATML_TPL}, f)
        seg_build, build_prompt, eot = cj.jinja_renderers(
            d, special_tokens={"bos_token": "", "eos_token": "<|im_end|>"})
        assert eot == "<|im_end|>"
        assert build_prompt("Q", system="S") == (
            "<|im_start|>system\nS<|im_end|>\n"
            "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n")
        assert build_prompt("Q") == (
            "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n")
        segs = seg_build([u("Q"), a("R")])
        assert sup(segs) == ["R<|im_end|>"]
    print("jinja_renderers: OK")


def test_derive_turn_close():
    # a template with no text after assistant content -> "" (not None)
    build = make_build(tpl="{%- for m in messages %}{{ m.content }}"
                           "{%- endfor %}")
    assert cj.derive_turn_close(build) == ""
    # a template that can't render the probe -> None (fallback to
    # turn_end_token in the trainers)
    build = make_build(tpl="{{ raise_exception('nope') }}")
    assert cj.derive_turn_close(build) is None
    print("derive_turn_close: OK")


if __name__ == "__main__":
    test_load_chat_template()
    test_single_turn()
    test_multi_turn()
    test_generation_prompt()
    test_nothink_header_split()
    test_reasoning_supervised()
    test_tool_flow()
    test_template_vars_layering()
    test_special_tokens_in_context()
    test_unrenderable()
    test_deferred_prefix()
    test_prefix_violation()
    test_multimodal_content_parts()
    test_extract_rich_turns()
    test_row_template_extras()
    test_jinja_renderers()
    test_derive_turn_close()
    print("\nALL OK")
