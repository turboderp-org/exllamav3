"""
CPU tests for training/chat_turns.py -- the multi-turn chat rendering +
loss-mask segmentation used by the SFT and preference trainers:

  * extract_turns normalization and single_turn_shape detection (the gate that
    keeps single-turn rows on the original, bit-for-bit-unchanged path);
  * per-format segment rendering: a [system?] user assistant conversation must
    concatenate to EXACTLY the string qlora_train_native.format_prompt_and_eot
    builds for that format, and only assistant contents (+ turn-end) may be
    supervised;
  * multi-turn rendering (separators, generation prompt, prior assistant
    turns), rejection of unrenderable shapes (tool role, mid-conversation
    system turn);
  * encode_segments / encode_completion BOS normalization and exact mask
    boundaries, via a fake tokenizer that auto-prepends BOS the way HF
    tokenizers do with add_bos_token=true.

No GPU / compiled extension / real model needed. Run:
    python tests/test_chat_turns.py
"""

from __future__ import annotations
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "chat_turns", os.path.join(_ROOT, "training", "chat_turns.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)


def u(c): return {"role": "user", "content": c}
def a(c): return {"role": "assistant", "content": c}
def s(c): return {"role": "system", "content": c}


def cat(segs): return "".join(t for t, _ in segs)
def sup(segs): return [t for t, is_sup in segs if is_sup]


# ---------------------------------------------------------------------------
# extract_turns / single_turn_shape / trim_trailing_context
# ---------------------------------------------------------------------------

def test_extract_turns():
    msgs = [{"role": "System", "content": " sys "},
            {"role": "USER", "content": "hi"},
            {"role": "assistant", "content": ""},        # empty -> dropped
            {"role": "assistant", "content": "yo "},
            {"content": "no role"},
            {"role": "tool", "content": "42"}]
    turns = ct.extract_turns(msgs)
    assert turns == [s("sys"), u("hi"), a("yo"), {"role": "", "content": "no role"},
                     {"role": "tool", "content": "42"}]
    assert ct.extract_turns(None) == []
    print("extract_turns: OK")


def test_single_turn_shape():
    assert ct.single_turn_shape([u("q"), a("r")]) == ("", "q", "r")
    assert ct.single_turn_shape([s("S"), u("q"), a("r")]) == ("S", "q", "r")
    assert ct.single_turn_shape([u("q")]) is None
    assert ct.single_turn_shape([u("q")], need_assistant=False) == ("", "q", "")
    assert ct.single_turn_shape([s("S"), u("q")], need_assistant=False) == ("S", "q", "")
    # anything beyond one exchange is NOT single-turn
    assert ct.single_turn_shape([u("a"), a("b"), u("c"), a("d")]) is None
    assert ct.single_turn_shape([u("a"), u("b"), a("c")]) is None
    assert ct.single_turn_shape([s("S"), u("q"), a("r")], need_assistant=False) is None
    assert ct.single_turn_shape([]) is None
    print("single_turn_shape: OK")


def test_trim_trailing_context():
    assert ct.trim_trailing_context([u("a"), a("b"), u("c")]) == [u("a"), a("b")]
    assert ct.trim_trailing_context([u("a"), a("b")]) == [u("a"), a("b")]
    assert ct.trim_trailing_context([u("a")]) == []
    print("trim_trailing_context: OK")


# ---------------------------------------------------------------------------
# Segment rendering
# ---------------------------------------------------------------------------

BOS, EOS = "<s>", "</s>"

# Expected single-turn strings, verbatim from format_prompt_and_eot's builders
# (prompt part) plus response + turn-end -- the equivalence that keeps the
# single-turn and multi-turn paths coherent.
SINGLE_TURN_EXPECTED = {
    "mistral": f"{BOS}[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][INST]Q[/INST]R{EOS}",
    "metharme": f"{BOS}<|system|>S<|user|>Q<|model|>R{EOS}",
    "gemma4-nothink": (f"{BOS}<|turn>system\nS<turn|>\n<|turn>user\nQ<turn|>\n"
                       f"<|turn>model\n<|channel>thought\n<channel|>R<turn|>"),
    "llama3": (f"{BOS}<|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>"
               f"<|start_header_id|>user<|end_header_id|>\n\nQ<|eot_id|>"
               f"<|start_header_id|>assistant<|end_header_id|>\n\nR<|eot_id|>"),
    "qwen3.5": ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
                "<|im_start|>assistant\nR<|im_end|>"),
    "qwen3.5-nothink": ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ"
                        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>"
                        "\n\nR<|im_end|>"),
    "chatml": ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
               "<|im_start|>assistant\nR<|im_end|>"),
    # HunYuan V3: the literal BOS of the model's own template (not the
    # tokenizer's bos_token argument), bare system text, no separators.
    "hy3": ("<｜hy_begin_of_sentence:opensource｜>S<｜hy_User:opensource｜>Q"
            "<｜hy_Assistant:opensource｜>R<｜hy_eos:opensource｜>"),
    "hy3-nothink": ("<｜hy_begin_of_sentence:opensource｜>S"
                    "<｜reasoning_mode:opensource｜>reasoning_effort:no_think"
                    "<｜hy_User:opensource｜>Q<｜hy_Assistant:opensource｜>"
                    "<think:opensource></think:opensource>R<｜hy_eos:opensource｜>"),
}

SINGLE_TURN_SUPERVISED = {
    "mistral": f"R{EOS}", "metharme": f"R{EOS}",
    "gemma4-nothink": "R<turn|>", "llama3": "R<|eot_id|>",
    "qwen3.5": "R<|im_end|>", "qwen3.5-nothink": "R<|im_end|>",
    "chatml": "R<|im_end|>",
    "hy3": "R<｜hy_eos:opensource｜>", "hy3-nothink": "R<｜hy_eos:opensource｜>",
}


def test_single_turn_equivalence():
    turns = [s("S"), u("Q"), a("R")]
    for fmt, expected in SINGLE_TURN_EXPECTED.items():
        build = ct.make_segment_builder(fmt, bos_token=BOS, eos_token=EOS)
        segs = build(turns)
        assert cat(segs) == expected, f"{fmt}: {cat(segs)!r} != {expected!r}"
        assert sup(segs) == [SINGLE_TURN_SUPERVISED[fmt]], f"{fmt}: {sup(segs)!r}"
        print(f"single-turn equivalence [{fmt}]: OK")


def test_multi_turn_llama3():
    build = ct.make_segment_builder("llama3", bos_token=BOS, eos_token=EOS)
    segs = build([u("Q1"), a("R1"), u("Q2"), a("R2")])
    assert cat(segs) == (
        f"{BOS}<|start_header_id|>user<|end_header_id|>\n\nQ1<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\nR1<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nQ2<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\nR2<|eot_id|>")
    assert sup(segs) == ["R1<|eot_id|>", "R2<|eot_id|>"]
    print("multi-turn [llama3]: OK")


def test_multi_turn_chatml_separator():
    # ChatML's user/system blocks carry their own trailing \n, but an assistant
    # close does not -- the \n before the next block belongs to (masked) context.
    build = ct.make_segment_builder("chatml", bos_token=BOS, eos_token=EOS)
    segs = build([u("Q1"), a("R1"), u("Q2"), a("R2")])
    assert cat(segs) == (
        "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nR1<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nR2<|im_end|>")
    assert sup(segs) == ["R1<|im_end|>", "R2<|im_end|>"]   # \n NOT supervised
    print("multi-turn [chatml] separator: OK")


def test_multi_turn_gemma4_separator():
    build = ct.make_segment_builder("gemma4-nothink", bos_token=BOS, eos_token=EOS)
    segs = build([u("Q1"), a("R1"), u("Q2"), a("R2")])
    assert cat(segs) == (
        f"{BOS}<|turn>user\nQ1<turn|>\n<|turn>model\n<|channel>thought\n"
        f"<channel|>R1<turn|>\n<|turn>user\nQ2<turn|>\n<|turn>model\n"
        f"<|channel>thought\n<channel|>R2<turn|>")
    assert sup(segs) == ["R1<turn|>", "R2<turn|>"]
    print("multi-turn [gemma4-nothink] separator: OK")


def test_multi_turn_hy3():
    # No separators between turns; the no_think header sits right after BOS
    # when there is no system turn (PromptFormat_hy3 with think=False), and
    # the empty think span is part of the MASKED assistant opener.
    build = ct.make_segment_builder("hy3-nothink", bos_token=BOS, eos_token=EOS)
    segs = build([u("Q1"), a("R1"), u("Q2"), a("R2")])
    assert cat(segs) == (
        "<｜hy_begin_of_sentence:opensource｜>"
        "<｜reasoning_mode:opensource｜>reasoning_effort:no_think"
        "<｜hy_User:opensource｜>Q1<｜hy_Assistant:opensource｜>"
        "<think:opensource></think:opensource>R1<｜hy_eos:opensource｜>"
        "<｜hy_User:opensource｜>Q2<｜hy_Assistant:opensource｜>"
        "<think:opensource></think:opensource>R2<｜hy_eos:opensource｜>")
    assert sup(segs) == ["R1<｜hy_eos:opensource｜>", "R2<｜hy_eos:opensource｜>"]
    build = ct.make_segment_builder("hy3", bos_token=BOS, eos_token=EOS)
    segs = build([u("Q1"), a("R1")], add_generation_prompt=False)
    assert cat(segs) == (
        "<｜hy_begin_of_sentence:opensource｜><｜hy_User:opensource｜>Q1"
        "<｜hy_Assistant:opensource｜>R1<｜hy_eos:opensource｜>")
    segs = build([u("Q1")], add_generation_prompt=True)
    assert cat(segs).endswith("Q1<｜hy_Assistant:opensource｜>")
    print("multi-turn [hy3 / hy3-nothink]: OK")


def test_generation_prompt():
    build = ct.make_segment_builder("chatml", bos_token=BOS, eos_token=EOS)
    # History ending in a user turn -- the TRL-conversational prompt shape.
    segs = build([u("Q1"), a("R1"), u("Q2")], add_generation_prompt=True)
    assert cat(segs) == (
        "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nR1<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n")
    # History ending in an assistant turn still owes the separator.
    segs = build([u("Q1"), a("R1")], add_generation_prompt=True)
    assert cat(segs).endswith("R1<|im_end|>\n<|im_start|>assistant\n")
    # Mistral's assistant opener is empty: the prompt just ends at [/INST].
    build = ct.make_segment_builder("mistral", bos_token=BOS, eos_token=EOS)
    segs = build([u("Q1")], add_generation_prompt=True)
    assert cat(segs) == f"{BOS}[INST]Q1[/INST]"
    print("generation prompt: OK")


def test_unrenderable_shapes():
    build = ct.make_segment_builder("chatml", bos_token=BOS, eos_token=EOS)
    for bad in ([u("q"), {"role": "tool", "content": "42"}, a("r")],
                [u("q"), s("late system"), a("r")],
                []):
        try:
            build(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    assert ct.make_segment_builder("auto") is None
    print("unrenderable shapes: OK")


# ---------------------------------------------------------------------------
# Tokenization: encode_segments / encode_completion
# ---------------------------------------------------------------------------

class _Ids(list):
    def tolist(self):
        return list(self)


class FakeTokenizer:
    """Char-level tokenizer that auto-prepends BOS on EVERY encode() call even
    with add_bos=False -- the HF add_bos_token=true behavior whose duplicates
    encode_segments/encode_completion must strip."""
    bos_token_id = 1

    def encode(self, text, add_bos=False, encode_special_tokens=True):
        return [_Ids([self.bos_token_id] + [ord(c) for c in text])]


def test_encode_segments():
    tok = FakeTokenizer()
    segs = [("ab", False), ("cd", True), ("e", False)]
    ids, labels = ct.encode_segments(tok, segs)
    # one BOS survives on the first segment; later auto-BOS are stripped
    assert ids == [1, ord("a"), ord("b"), ord("c"), ord("d"), ord("e")]
    assert labels == [-100, -100, -100, ord("c"), ord("d"), -100]
    # mask boundary is exact: supervised ids == the supervised segment alone
    supd = [t for t, l in zip(ids, labels) if l != -100]
    assert supd == [ord("c"), ord("d")]
    print("encode_segments: OK")


def test_encode_completion():
    tok = FakeTokenizer()
    ids = ct.encode_completion(tok, "hi", "!")
    assert ids == [ord("h"), ord("i"), ord("!")]   # BOS stripped, eot appended
    print("encode_completion: OK")


if __name__ == "__main__":
    test_extract_turns()
    test_single_turn_shape()
    test_trim_trailing_context()
    test_single_turn_equivalence()
    test_multi_turn_llama3()
    test_multi_turn_chatml_separator()
    test_multi_turn_gemma4_separator()
    test_multi_turn_hy3()
    test_generation_prompt()
    test_unrenderable_shapes()
    test_encode_segments()
    test_encode_completion()
    print("\nALL OK")
