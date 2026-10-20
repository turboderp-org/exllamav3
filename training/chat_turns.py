"""
Multi-turn chat rendering with exact loss-mask segmentation.

The single-turn pipeline (qlora_train_native.build_sft_examples /
qlora_train_pref) builds one prompt string and one response string, tokenizes
them SEPARATELY and masks the prompt to -100. That is exact but can only
express one prompt/response boundary per sequence. This module generalizes it:
a conversation is rendered as an ordered list of ``(text, supervised)``
SEGMENTS -- one or more per turn -- which are tokenized separately and
concatenated, so every user/assistant boundary in a multi-turn conversation is
an exact token boundary in the mask.

Supervision policy: ONLY assistant turn contents (plus their turn-end token)
are supervised; system/user turns and the assistant turn headers are masked to
-100. Training on user text teaches the model to imitate users and continue
prompts, so completion-only is the default and only mode -- same as the
single-turn path, TRL's trainers and Unsloth's train_on_responses_only.

This is deliberately NOT the Unsloth approach of rendering the whole
conversation through the chat template and then string-searching for
``instruction_part``/``response_part`` markers: segment rendering needs no
knowledge of what a Jinja template inserts, can't mis-match markers that
appear inside message contents, and rejects roles it can't render (``tool``)
instead of silently mislabeling them. (``--prompt-format jinja`` --
training/chat_jinja.py -- renders through the model's own Jinja template
instead, emitting this same segment contract via incremental prefix
rendering; it handles tool roles / tool_calls / reasoning_content that these
hardcoded formats reject.)

``--prompt-format auto`` (model.default_chat_prompt) has no multi-turn
renderer -- the per-arch default prompts are single-turn by construction --
so callers fail fast with AUTO_SINGLE_TURN_HINT rather than silently
truncating the conversation.

Pure python, no imports -- unit-testable without torch / the compiled
extension (tests/test_chat_turns.py).
"""

SEGMENT_FORMATS = ("mistral", "metharme", "gemma4-nothink", "llama3",
                   "qwen3.5", "qwen3.5-nothink", "chatml", "hy3", "hy3-nothink")

# HunYuan V3 (HYV3ForCausalLM) control tokens. The template writes BOS
# literally (the model's default_chat_prompt and PromptFormat_hy3 in
# examples/chat_templates.py both do; the tokenizer's own BOS is the same
# token, so the encoder's single-BOS normalization collapses any duplicate).
HY3_BOS = "<｜hy_begin_of_sentence:opensource｜>"
HY3_USER = "<｜hy_User:opensource｜>"
HY3_ASSISTANT = "<｜hy_Assistant:opensource｜>"
HY3_EOS = "<｜hy_eos:opensource｜>"
HY3_NOTHINK_HEADER = "<｜reasoning_mode:opensource｜>reasoning_effort:no_think"
HY3_EMPTY_THINK = "<think:opensource></think:opensource>"

AUTO_SINGLE_TURN_HINT = (
    "--prompt-format auto renders single-turn prompts only "
    "(model.default_chat_prompt); this dataset has multi-turn conversations. "
    "Pass an explicit --prompt-format (" + " / ".join(SEGMENT_FORMATS) + ", "
    "or jinja to use the model directory's own chat template) so each turn "
    "can be rendered and loss-masked individually.")


def extract_turns(messages):
    """Normalize an OpenAI-style ``messages`` list to
    ``[{"role", "content"}, ...]``: roles lowercased, contents stripped,
    empty-content turns dropped. No role validation here -- the segment
    builder rejects roles it can't render (e.g. ``tool``) so the caller can
    skip and count the row."""
    turns = []
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if content:
            turns.append({"role": role, "content": content})
    return turns


def turn_text(turn):
    """Plain-text view of a (possibly rich) turn's content: the string
    itself, or the concatenated text parts of a multimodal content-parts
    list ([{"type": "text", "text": ...}, {"type": "image", ...}, ...]).
    Used for word counts / cleaning on rows the jinja path renders."""
    c = turn.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)):
        return " ".join(
            (p.get("text") or "") for p in c
            if isinstance(p, dict) and (p.get("type") or "text") == "text")
    return ""


def single_turn_shape(turns, need_assistant=True):
    """Return ``(sys_text, user_text, asst_text)`` when ``turns`` is exactly
    the shape the original single-turn pipeline supported -- ``[system?] user
    assistant`` (or ``[system?] user`` with ``need_assistant=False``) -- else
    None. Callers keep these rows on the pre-existing single-string prompt
    path so single-turn tokenization (and hence existing runs) stays
    bit-for-bit unchanged; only genuine multi-turn rows take the segment
    renderer."""
    if turns and turns[0]["role"] == "system":
        sys_text, rest = turns[0]["content"], turns[1:]
    else:
        sys_text, rest = "", turns
    roles = [t["role"] for t in rest]
    want = ["user", "assistant"] if need_assistant else ["user"]
    if roles != want:
        return None
    asst = rest[1]["content"] if need_assistant else ""
    return sys_text, rest[0]["content"], asst


def trim_trailing_context(turns):
    """Drop turns after the last assistant turn -- nothing supervises them,
    so they would only burn sequence budget as dead context."""
    last = -1
    for i, t in enumerate(turns):
        if t["role"] == "assistant":
            last = i
    return turns[:last + 1]


def make_segment_builder(prompt_format, bos_token="", eos_token=""):
    """Return ``build(turns, add_generation_prompt=False) -> [(text,
    supervised)]`` for an explicit prompt format, or None for formats without
    a multi-turn renderer (``auto``).

    The piece templates mirror qlora_train_native.format_prompt_and_eot
    exactly, so a ``[system?] user assistant`` conversation concatenates to
    the same text the single-turn builder emits (the single-turn path is still
    used for those rows; the equivalence is what keeps the two paths
    coherent). ``build`` raises ValueError on shapes it can't render (a role
    with no defined rendering, a system turn that isn't first) so callers can
    skip and count the row.

    ``add_generation_prompt=True`` appends the (masked) assistant turn opener,
    for rendering a preference-data prompt/history that a separately-encoded
    completion will follow."""
    bos = bos_token or ""
    eos = eos_token or ""
    head_tail = ""   # text after the system turn, system or not (hy3-nothink)
    if prompt_format == "mistral":
        sys_fmt, user_fmt = "[SYSTEM_PROMPT]{}[/SYSTEM_PROMPT]", "[INST]{}[/INST]"
        asst_open, close, sep = "", eos, ""
    elif prompt_format == "metharme":
        sys_fmt, user_fmt = "<|system|>{}", "<|user|>{}"
        asst_open, close, sep = "<|model|>", eos, ""
    elif prompt_format == "gemma4-nothink":
        sys_fmt = "<|turn>system\n{}<turn|>\n"
        user_fmt = "<|turn>user\n{}<turn|>\n"
        asst_open = "<|turn>model\n<|channel>thought\n<channel|>"
        close, sep = "<turn|>", "\n"
    elif prompt_format == "llama3":
        sys_fmt = "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>"
        user_fmt = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        asst_open = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        close, sep = "<|eot_id|>", ""
    elif prompt_format in ("qwen3.5", "qwen3.5-nothink", "chatml"):
        nothink = "<think>\n\n</think>\n\n" if prompt_format.endswith("-nothink") else ""
        sys_fmt = "<|im_start|>system\n{}<|im_end|>\n"
        user_fmt = "<|im_start|>user\n{}<|im_end|>\n"
        asst_open = "<|im_start|>assistant\n" + nothink
        close, sep = "<|im_end|>", "\n"
        bos = ""   # ChatML formats prepend no BOS (matches format_prompt_and_eot)
    elif prompt_format in ("hy3", "hy3-nothink"):
        # HunYuan V3: BOS, the bare system text, then <｜hy_User｜>/<｜hy_Assistant｜>
        # turn markers with no separators; <｜hy_eos｜> ends a turn. "-nothink"
        # is PromptFormat_hy3 with thinking off: a reasoning_effort:no_think
        # header right after the (optional) system text and the think span
        # pre-closed empty in the masked assistant opener.
        nothink = prompt_format.endswith("-nothink")
        bos = HY3_BOS
        sys_fmt, user_fmt = "{}", HY3_USER + "{}"
        head_tail = HY3_NOTHINK_HEADER if nothink else ""
        asst_open = HY3_ASSISTANT + (HY3_EMPTY_THINK if nothink else "")
        close, sep = HY3_EOS, ""
    else:
        return None

    def build(turns, add_generation_prompt=False):
        if not turns:
            raise ValueError("empty conversation")
        segs = []
        rest = list(turns)
        head = bos
        if rest[0]["role"] == "system":
            head += sys_fmt.format(rest[0]["content"])
            rest = rest[1:]
        head += head_tail
        if head:
            segs.append((head, False))
        after_asst = False   # a supervised close just ended; sep may be owed
        for t in rest:
            role, content = t["role"], t["content"]
            lead = sep if after_asst else ""
            if role == "user":
                segs.append((lead + user_fmt.format(content), False))
                after_asst = False
            elif role == "assistant":
                if lead or asst_open:
                    segs.append((lead + asst_open, False))
                segs.append((content + close, True))
                after_asst = True
            elif role == "system":
                raise ValueError("system turn not at the start of the conversation")
            else:
                raise ValueError(
                    f"role '{role}' has no rendering in '{prompt_format}' "
                    f"(system/user/assistant only)")
        if add_generation_prompt:
            lead = sep if after_asst else ""
            if lead or asst_open:
                segs.append((lead + asst_open, False))
        return segs

    return build


def encode_segments(tokenizer, segments):
    """Tokenize ``(text, supervised)`` segments SEPARATELY and concatenate to
    ``(input_ids, labels)`` int lists, labels -100 on unsupervised segments.
    Separate encoding keeps every mask boundary exact (no tokenizer merges
    across a boundary); BOS is normalized to exactly one leading token, the
    same way encode_prompt_response does it (the HF tokenizer may auto-prepend
    one per encode() call even with add_bos=False)."""
    bos = tokenizer.bos_token_id
    input_ids, labels = [], []
    for text, supervised in segments:
        ids = tokenizer.encode(
            text, add_bos=False, encode_special_tokens=True)[0].tolist()
        if bos is not None and ids:
            if not input_ids:
                while len(ids) >= 2 and ids[0] == bos and ids[1] == bos:
                    ids = ids[1:]
            elif ids[0] == bos:
                ids = ids[1:]
        input_ids += ids
        labels += list(ids) if supervised else [-100] * len(ids)
    return input_ids, labels


def encode_completion(tokenizer, text, eot):
    """Tokenize a bare completion (+ turn-end token) that will follow a
    separately-encoded prompt: no BOS, and any auto-prepended one stripped
    (same normalization as encode_prompt_response's response side)."""
    ids = tokenizer.encode(
        text + eot, add_bos=False, encode_special_tokens=True)[0].tolist()
    bos = tokenizer.bos_token_id
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return ids
