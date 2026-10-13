"""
Jinja chat-template rendering with exact loss-mask segmentation
(``--prompt-format jinja``).

Renders conversations through the model's OWN chat template -- the same Jinja
template inference servers apply -- instead of the hardcoded formats in
training/chat_turns.py, and emits the same ordered ``[(text, supervised)]``
segment contract ``make_segment_builder`` produces, so build_sft_examples and
the preference builders consume it unchanged. What the hardcoded formats can't
express, the template can: ``tool`` roles, ``tool_calls`` and
``reasoning_content`` on assistant messages, top-level ``tools`` schemas, and
template variables like ``enable_thinking`` (rows carry them under
``template_vars`` / ``chat_template_kwargs`` -- tabby's and llama-server's
names for the same bag; both are accepted).

Segmentation is INCREMENTAL RENDERING: the template is applied to every
``messages[:i]`` prefix (always with ``add_generation_prompt=False``) and each
turn's text is the diff between consecutive renders, so no marker
string-search ever touches message contents. For an assistant turn the diff is
split into a masked header and a supervised body at the longest common prefix
with the ``add_generation_prompt=True`` render of the same history -- that is
exactly the text the template would prefill at inference (e.g. Qwen's
``<|im_start|>assistant\\n`` plus the empty ``<think>`` block when
``enable_thinking`` is false), so headers/prefills are masked and everything
the model would actually generate (reasoning span, content, tool-call block,
turn-end token) is supervised. Trailing whitespace after the turn-end token is
carried into the NEXT masked segment (the inter-turn separator convention the
hardcoded formats use).

This requires the template to be PREFIX-MONOTONIC: rendering ``messages[:i+1]``
must extend the ``messages[:i]`` rendering verbatim. Templates that re-render
history -- stripping ``reasoning_content`` from earlier assistant turns (Qwen3
thinking), or injecting ``tools`` before the LAST user turn (Mistral V7) --
violate this for rows that exercise those paths; such rows raise ValueError
and are skipped and counted by the callers, with the reason in the message.
Rows that keep reasoning only on the final assistant turn (what OAI-style APIs
return anyway) render fine under those same templates.

Multimodal NOTE: list-form message content (OpenAI content-parts,
``[{"type": "text", ...}, {"type": "image", ...}]``) passes through to the
template untouched, so multimodal-formatted datasets render -- but this
trainer is text-only: image/video/audio parts become the template's
placeholder tokens with no pixel features attached. Callers count such rows
(:func:`has_nontext_parts`) and tell the user; only the text is trained.

jinja2 is the only dependency (a torch dependency already, so always
importable in a working venv) and it is imported lazily -- importing this
module costs nothing. Unit-testable without torch / the compiled extension
(tests/test_chat_jinja.py).
"""

import datetime
import json
import os
import re


# ---------------------------------------------------------------------------
# Template loading (from the model directory, the way HF transformers does)
# ---------------------------------------------------------------------------

def load_chat_template(model_dir, template_file=None, name=None):
    """Resolve the Jinja chat-template source for a model directory, in the
    same priority order HF transformers uses: an explicit override file wins;
    then ``chat_template.jinja`` (the single-file convention newer exports
    use); then ``chat_template.json`` (the multimodal-processor convention);
    then the ``"chat_template"`` key of ``tokenizer_config.json``. The stored
    value may be a plain template string or a list of ``{"name", "template"}``
    dicts (named templates) -- ``name`` picks one, default ``"default"``.
    Raises ValueError when no template is found anywhere."""
    if template_file:
        with open(os.path.expanduser(template_file), encoding="utf-8") as f:
            return f.read()
    p = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    for fn in ("chat_template.json", "tokenizer_config.json"):
        p = os.path.join(model_dir, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            tpl = json.load(f).get("chat_template")
        if tpl is not None:
            return _pick_named_template(tpl, name, p)
    raise ValueError(
        f"no chat template found in {model_dir} (looked for "
        f"chat_template.jinja, chat_template.json, and the 'chat_template' "
        f"key of tokenizer_config.json). Pass --chat-template-file to point "
        f"at a Jinja template, or use one of the hardcoded --prompt-format "
        f"options.")


def _pick_named_template(tpl, name, path):
    """A stored chat_template is either the template string itself or a list
    of {"name", "template"} dicts (e.g. Cohere's default/tool_use/rag)."""
    if isinstance(tpl, str):
        return tpl
    if isinstance(tpl, list):
        want = name or "default"
        by_name = {t.get("name"): t.get("template") for t in tpl
                   if isinstance(t, dict)}
        if want in by_name and by_name[want]:
            return by_name[want]
        raise ValueError(
            f"{path} holds named chat templates "
            f"({', '.join(sorted(k for k in by_name if k))}) but none called "
            f"{want!r}")
    raise ValueError(f"unrecognized chat_template value in {path}: "
                     f"{type(tpl).__name__}")


# ---------------------------------------------------------------------------
# Compilation (HF-compatible sandboxed environment)
# ---------------------------------------------------------------------------

def compile_chat_template(source):
    """Compile a template source to ``render(messages,
    add_generation_prompt=False, **vars) -> str``. The environment matches
    what HF apply_chat_template runs templates in -- ImmutableSandboxed,
    trim_blocks/lstrip_blocks, loopcontrols, and the ``tojson`` /
    ``raise_exception`` / ``strftime_now`` helpers -- so templates behave
    exactly as they do at inference. Render-time template errors (including
    ``raise_exception`` calls and type errors from message-shape mismatches)
    surface as ValueError so callers can skip and count the row instead of
    crashing the run."""
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as e:
        raise SystemExit(
            f"--prompt-format jinja needs the jinja2 package, which is not "
            f"importable ({e}). Install it in this venv (pip install jinja2).")

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = (
        lambda x, ensure_ascii=False, indent=None, separators=None,
        sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii,
                                    indent=indent, separators=separators,
                                    sort_keys=sort_keys))
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = (
        lambda fmt: datetime.datetime.now().strftime(fmt))
    try:
        template = env.from_string(source)
    except jinja2.exceptions.TemplateError as e:
        raise ValueError(f"chat template failed to compile: {e}") from e

    def render(messages, add_generation_prompt=False, **kwargs):
        try:
            return template.render(messages=messages,
                                   add_generation_prompt=add_generation_prompt,
                                   **kwargs)
        except jinja2.exceptions.TemplateError as e:
            raise ValueError(f"chat template failed to render: {e}") from e
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            raise ValueError(
                f"chat template failed to render ({type(e).__name__}: {e}); "
                f"the message shapes probably don't match what the template "
                f"expects") from e

    return render


# ---------------------------------------------------------------------------
# Optional system-preamble scrubbing (--strip-sys-prompt-extras)
# ---------------------------------------------------------------------------

# Boilerplate that harmony-style templates (Muse-Glimmer, gpt-oss and friends)
# synthesize into the system block when a row has no system message of its own.
# None of it comes from the dataset, and for a persona/voice fine-tune all three
# are actively unhelpful:
#
#   Reasoning strength: <level>   the macro default is the model's HIGHEST or a
#                                 fixed level, and if the training answers carry
#                                 no reasoning channel the run teaches "this
#                                 effort level => emit no reasoning"
#   Knowledge cutoff: <date>      emitted unconditionally by some templates; no
#                                 template variable suppresses it
#   Current date: <date>          falls through to strftime_now() when the
#                                 caller passes no current_date, so the rendered
#                                 corpus silently changes from day to day and
#                                 runs stop being byte-reproducible
#
# Only some of these are reachable via --template-vars, and only per-call-site,
# which is why this is a render-level scrub rather than a set of variables.
SYS_PROMPT_EXTRA_LINE = re.compile(
    r"(?m)^(?:Reasoning strength|Knowledge cutoff|Current date)[ \t]*:[^\n]*\n?")


_STRIP_EXTRAS_DEFAULT = False


def set_strip_sys_prompt_extras(on):
    """Process-wide default for jinja_renderers(strip_extras=...). The trainers
    call this once from their arg parsing so the flag reaches every renderer
    without threading a parameter through build_dataset/format_prompt_and_eot
    and their callers in four separate scripts."""
    global _STRIP_EXTRAS_DEFAULT
    _STRIP_EXTRAS_DEFAULT = bool(on)


def strip_sys_prompt_extras(text):
    """Delete template-injected reasoning-strength / knowledge-cutoff /
    current-date lines from a rendered prompt.

    Removing whole lines leaves a longer run of newlines behind where the block
    used to be, so runs of 3+ newlines collapse back to a single blank line --
    which restores exactly the spacing the template would have produced had the
    lines never been emitted. That collapse is global, so a *message* whose own
    content contains three or more consecutive newlines would be normalized too;
    keep that in mind for corpora with ASCII art or deliberate vertical space.
    """
    out = SYS_PROMPT_EXTRA_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", out)


# ---------------------------------------------------------------------------
# Message / row normalization
# ---------------------------------------------------------------------------

def extract_rich_turns(messages):
    """Normalize an OpenAI-style ``messages`` list for Jinja rendering,
    KEEPING the rich keys the hardcoded formats can't express:
    ``reasoning_content``, ``tool_calls``, ``tool_call_id``, ``name``, and
    list-form (multimodal content-parts) content. Roles are lowercased,
    string contents stripped, and None values dropped (HF datasets' Arrow
    schema unification pads absent fields with None on every message).
    ``tool_calls`` are normalized to the shape templates expect --
    ``{"type": "function", "function": {"name", "arguments"}}`` with
    ``arguments`` parsed from the OAI wire-format JSON string to a dict, so
    templates that do ``tool_call.arguments | tojson`` don't double-encode.
    Turns with nothing to render (no content, no reasoning_content, no
    tool_calls) are dropped, EXCEPT that an assistant turn's empty content
    survives alongside its tool_calls."""
    turns = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        t = {k: v for k, v in m.items() if v is not None and k != "role"}
        t["role"] = (m.get("role") or "").lower()
        content = t.get("content")
        if isinstance(content, str):
            content = content.strip()
        t["content"] = content if content is not None else ""
        if isinstance(t.get("reasoning_content"), str):
            t["reasoning_content"] = t["reasoning_content"].strip()
        if not t.get("reasoning_content"):
            t.pop("reasoning_content", None)
        tcs = [_norm_tool_call(tc) for tc in t.get("tool_calls") or []]
        if tcs:
            t["tool_calls"] = tcs
        else:
            t.pop("tool_calls", None)
        if t["content"] or t.get("tool_calls") or t.get("reasoning_content"):
            turns.append(t)
    return turns


def _norm_tool_call(tc):
    """One tool call -> the nested HF shape templates expect. Accepts the
    flat legacy shape (name/arguments at the top level) and JSON-string
    arguments; anything unrecognized passes through untouched."""
    if not isinstance(tc, dict):
        return tc
    tc = {k: v for k, v in tc.items() if v is not None}
    fn = tc.get("function")
    if not isinstance(fn, dict):
        fn = {k: tc.pop(k) for k in ("name", "arguments") if k in tc}
    fn = {k: v for k, v in fn.items() if v is not None}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            fn["arguments"] = json.loads(args)
        except ValueError:
            pass   # not JSON; leave the string for the template to handle
    tc["type"] = tc.get("type") or "function"
    tc["function"] = fn
    return tc


def _maybe_json(v):
    """Arrow columns holding heterogeneous dicts/lists often arrive as JSON
    strings; parse those, pass real values through, drop unparseable text."""
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def row_template_extras(row):
    """Per-row template inputs from a dataset row: ``tools`` (next to
    ``messages``, the OAI convention) plus the ``template_vars`` /
    ``chat_template_kwargs`` bags (synonyms; ``template_vars`` wins on key
    conflicts). Returns ``{"tools": ..., "template_vars": ...}`` kwargs for a
    segment ``build()`` call. Keys that would collide with the render call
    itself (messages / add_generation_prompt) are dropped; a ``tools`` key
    inside a var bag is honored when the row has no top-level ``tools``."""
    tvars = {}
    for key in ("chat_template_kwargs", "template_vars"):
        v = _maybe_json(row.get(key))
        if isinstance(v, dict):
            tvars.update(v)
    tools = _maybe_json(row.get("tools"))
    if tools is None:
        tools = tvars.pop("tools", None)
    else:
        tvars.pop("tools", None)
    for k in ("messages", "add_generation_prompt"):
        tvars.pop(k, None)
    return {"tools": tools, "template_vars": tvars or None}


def has_nontext_parts(turns):
    """True when any message's content is a content-parts list holding a
    non-text part (image / video / audio / ...). Such rows still render --
    the template emits its placeholder tokens -- but this trainer attaches no
    pixel features, so callers count them and say so."""
    for t in turns:
        c = t.get("content")
        if isinstance(c, (list, tuple)):
            for p in c:
                if isinstance(p, dict) and (p.get("type") or "text") != "text":
                    return True
    return False


# ---------------------------------------------------------------------------
# Segment building (incremental rendering)
# ---------------------------------------------------------------------------

def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def make_jinja_segment_builder(render, default_vars=None):
    """Return ``build(turns, add_generation_prompt=False, tools=None,
    template_vars=None) -> [(text, supervised)]`` -- the make_segment_builder
    contract, driven by a compiled template's ``render`` instead of hardcoded
    piece strings. ``default_vars`` (special tokens, CLI --template-vars) are
    in every render's context, with per-call ``template_vars`` layered on
    top. See the module docstring for the incremental-rendering algorithm and
    the prefix-monotonicity requirement; violations raise ValueError so
    callers skip and count the row."""
    base_vars = dict(default_vars or {})

    def build(turns, add_generation_prompt=False, tools=None,
              template_vars=None):
        if not turns:
            raise ValueError("empty conversation")
        ctx = dict(base_vars)
        if template_vars:
            ctx.update(template_vars)
        for k in ("messages", "add_generation_prompt"):
            ctx.pop(k, None)
        if tools is None:
            tools = ctx.pop("tools", None)
        else:
            ctx.pop("tools", None)

        def r(msgs, gen=False):
            return render(list(msgs), add_generation_prompt=gen, tools=tools,
                          **ctx)

        segs, prev, pending = [], "", ""
        for i, t in enumerate(turns):
            if t.get("role") != "assistant" and i + 1 < len(turns):
                # Some templates refuse a prefix with no user turn yet
                # (Qwen3.6/3.8: raise_exception('No user query found') on a
                # lone system message). Such a prefix is masked context
                # either way, so defer it: the next turn's render covers
                # both and the diff lands in one masked segment. Only the
                # LAST turn's render has to succeed on its own.
                try:
                    cur = r(turns[:i + 1])
                except ValueError:
                    continue
            else:
                cur = r(turns[:i + 1])
            if not cur.startswith(prev):
                raise ValueError(
                    f"template re-renders earlier turns at turn {i} "
                    f"({t.get('role')!r}): messages[:{i + 1}] does not extend "
                    f"the messages[:{i}] rendering, so exact per-turn masks "
                    f"can't be cut. Common causes: the template strips "
                    f"reasoning_content from history (keep reasoning only on "
                    f"the final assistant turn) or injects tools before the "
                    f"last user turn.")
            if t.get("role") == "assistant":
                # The add_generation_prompt render of the same history is
                # exactly the header/prefill the template would put in front
                # of a model reply -- everything up to where it and the full
                # turn diverge is masked, the rest (reasoning span, content,
                # tool-call block, turn close) is supervised.
                try:
                    with_gen = r(turns[:i], gen=True)
                except ValueError as e:
                    raise ValueError(
                        f"can't split the assistant header at turn {i}: the "
                        f"add_generation_prompt rendering of the preceding "
                        f"history failed ({e})")
                split = max(_common_prefix_len(cur, with_gen), len(prev))
                header, body = cur[len(prev):split], cur[split:]
                sup = body.rstrip()
                if not sup:
                    raise ValueError(
                        f"assistant turn {i} rendered empty (nothing to "
                        f"supervise)")
                if pending or header:
                    segs.append((pending + header, False))
                segs.append((sup, True))
                pending = body[len(sup):]
            else:
                seg = pending + cur[len(prev):]
                if seg:
                    segs.append((seg, False))
                pending = ""
            prev = cur
        if add_generation_prompt:
            final = r(turns, gen=True)
            if not final.startswith(prev):
                raise ValueError(
                    "the add_generation_prompt rendering does not extend the "
                    "conversation rendering (the template restructures "
                    "history when adding the generation prompt)")
            seg = pending + final[len(prev):]
            if seg:
                segs.append((seg, False))
        elif pending:
            segs.append((pending, False))
        return segs

    return build


_PROBE = "EXL3JINJAPROBE"


def derive_turn_close(build):
    """The text a template appends after assistant content -- the turn-end
    ('eot') string the trainers append to bare completions and check
    truncation against. Derived by segment-rendering a probe conversation and
    slicing the supervised text after the sentinel; trailing whitespace is
    already excluded (the builder masks it as inter-turn separator). Returns
    None when the template can't render the probe (callers fall back to
    turn_end_token())."""
    try:
        segs = build([{"role": "user", "content": "probe"},
                      {"role": "assistant", "content": _PROBE}])
    except ValueError:
        return None
    sup = "".join(t for t, s in segs if s)
    i = sup.rfind(_PROBE)
    if i < 0:
        return None
    return sup[i + len(_PROBE):]


# ---------------------------------------------------------------------------
# Trainer-facing conveniences
# ---------------------------------------------------------------------------

def tokenizer_special_tokens(tokenizer):
    """The special-token context HF apply_chat_template exposes to templates
    ({{ bos_token }} etc.), pulled off an exllamav3 Tokenizer."""
    return {k: getattr(tokenizer, k, None) or ""
            for k in ("bos_token", "eos_token", "pad_token", "unk_token")}


def parse_template_vars(text):
    """--template-vars CLI value (a JSON object, e.g.
    '{"enable_thinking": false}') -> dict or None."""
    if not text:
        return None
    try:
        v = json.loads(text)
    except ValueError as e:
        raise SystemExit(f"--template-vars is not valid JSON: {e}")
    if not isinstance(v, dict):
        raise SystemExit("--template-vars must be a JSON object "
                         "(e.g. '{\"enable_thinking\": false}')")
    return v


def jinja_renderers(model_dir, special_tokens=None, template_file=None,
                    default_vars=None, template_name=None,
                    strip_extras=None):
    """One-stop factory for the trainers: load + compile the model's chat
    template and return ``(seg_build, build_prompt, eot)``:

      * ``seg_build(turns, add_generation_prompt=False, tools=None,
        template_vars=None)`` -> ``[(text, supervised)]`` segments;
      * ``build_prompt(user, system=None)`` -> single-turn prompt string
        ending with the generation prompt (the format_prompt_and_eot
        contract, for the flat-column and --sample-every paths);
      * ``eot``: the probe-derived turn-close string, or None when the
        template can't render the probe (use turn_end_token() then).

    ``special_tokens`` and ``default_vars`` (e.g. {"enable_thinking": False})
    are in every render's context; per-row template_vars layer on top.

    ``strip_extras`` (--strip-sys-prompt-extras) wraps the compiled renderer in
    strip_sys_prompt_extras(), so the scrub happens once, below every consumer:
    seg_build, build_prompt and the derive_turn_close probe all see the same
    text and the segment prefix-matching stays consistent with it. Defaults to
    whatever set_strip_sys_prompt_extras() last set (off unless a trainer opts
    in), so behaviour is unchanged for every existing caller."""
    source = load_chat_template(model_dir, template_file, name=template_name)
    render = compile_chat_template(source)
    if strip_extras is None:
        strip_extras = _STRIP_EXTRAS_DEFAULT
    if strip_extras:
        _raw_render = render

        def render(messages, add_generation_prompt=False, **kwargs):
            return strip_sys_prompt_extras(
                _raw_render(messages,
                            add_generation_prompt=add_generation_prompt,
                            **kwargs))
    base = dict(special_tokens or {})
    base.update(default_vars or {})
    seg_build = make_jinja_segment_builder(render, base)

    def build_prompt(user, system=None):
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": user})
        return "".join(t for t, _ in
                       seg_build(msgs, add_generation_prompt=True))

    return seg_build, build_prompt, derive_turn_close(seg_build)
