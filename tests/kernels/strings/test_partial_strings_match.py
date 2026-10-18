"""
partial_strings_match (generator/strings.cpp): scans the held text Q for any of the stop/banned strings S.
Contract used by job.py: >= 0 -> index of the EARLIEST full match (text is truncated there); -2 -> some string
partially overlaps the end of Q and starts before any full match (hold text, wait for more); -1 -> nothing.
Regression: a partial match of an earlier-listed string used to short-circuit before a later string's full
match at a smaller index was found, so an already-present stop string could be overrun. Checked against
hand-computed expectations. Empty inputs: an empty Q or an empty set of strings (no offsets, or a lone 0 offset)
matches nothing (-1).
"""

import numpy as np
import pytest

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.generator.job import _strings_to_utf32

pytestmark = pytest.mark.nogpu


def match(q, strings):
    buf, offs = _strings_to_utf32(tuple(strings))
    return ext.partial_strings_match(np.frombuffer(q.encode("utf-32-le"), dtype = np.uint8), offs, buf)


def test_basic():
    assert match("hello world", ["world"]) == 6
    assert match("hello world", ["xyz"]) == -1
    assert match("hello wor", ["world"]) == -2
    assert match("", ["a"]) == -1


def test_earliest_full_match_across_strings():
    # full match of a later-listed string at a smaller index must win over an earlier-listed string's match
    assert match("STOP and then END", ["END", "STOP"]) == 0
    assert match("aaXbbYcc", ["Y", "X"]) == 2


def test_full_match_not_hidden_by_later_partial():
    # "<|im_end|>" partially overlaps the tail ("<|im"), but "STOP" is already fully present earlier: must stop at 0
    assert match("STOP text <|im", ["<|im_end|>", "STOP"]) == 0
    assert match("STOP text <|im", ["STOP", "<|im_end|>"]) == 0


def test_partial_before_full_holds():
    # a partial that starts before the earliest full match may still turn into an earlier stop: wait
    assert match("ab STOP", ["abc", "STOP"]) == 3          # "ab" does not reach the end: no partial
    assert match("xx ab", ["abc", "xx"]) == 0               # full at 0 precedes the partial at 3
    assert match("ab xx", ["abcdef", "xx"]) == 3            # partial "ab"? no: "ab" is followed by " " so it fails; full "xx" at 3
    assert match("q ab", ["abcd", "zzz"]) == -2             # only a trailing partial


def test_partial_earlier_than_full_waits():
    # partial "ab…" at index 0 could complete into an earlier stop than the full "b" at 1: must hold
    assert match("ab", ["abc", "b"]) == -2


def test_multiple_and_empty_strings():
    assert match("one two three", ["three", "two", ""]) == 4
    assert match("abcabc", ["cab", "abc"]) == 0
    assert match("abcab", ["cab", "abcabc"]) == -2          # full "cab" at 2 but partial "abcab…" at 0 starts earlier -> hold



@pytest.mark.parametrize("q", ["", "abc"])
def test_empty_strings_match(q):
    if not q:
        assert match(q, ["a"]) == -1
        assert match(q, ["a", "bc"]) == -1
    qb = np.frombuffer(q.encode("utf-32-le"), dtype = np.uint8)
    empty = np.zeros(0, dtype = np.uint8)
    lone = np.frombuffer(np.array([0], dtype = np.int32), dtype = np.uint8)
    assert ext.partial_strings_match(qb, empty, empty) == -1
    assert ext.partial_strings_match(qb, lone, empty) == -1
