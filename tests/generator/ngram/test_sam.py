"""
BC_SAM suffix automaton (the live n-gram index): accept() returns the span of the longest earlier occurrence of
the current suffix, through promoted hub states, clones, sparse/negative token IDs, resets, chunked
accept_tensor() calls and rewinds. Checked against a dictionary-transition reference SAM and brute-force
suffix search.
"""

import random

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.nogpu


class ReferenceSAM:
    """Dictionary transitions, independent of the C++ edge/index representation."""
    def __init__(self):
        self.states = [dict(link = -1, length = 0, end = -1, edges = {})]
        self.last = self.match = self.match_len = self.pos = 0

    def accept(self, token):
        s = self.states
        while self.match and token not in s[self.match]["edges"]:
            self.match = s[self.match]["link"]
            self.match_len = min(self.match_len, s[self.match]["length"])
        if token in s[self.match]["edges"]:
            self.match = s[self.match]["edges"][token]
            self.match_len += 1
        else:
            self.match = self.match_len = 0
        end = s[self.match]["end"] + 1
        result = (end - self.match_len, end) if self.match_len else (-1, -1)
        cur = len(s)
        s.append(dict(link = 0, length = s[self.last]["length"] + 1, end = self.pos, edges = {}))
        p = self.last
        while p != -1 and token not in s[p]["edges"]:
            s[p]["edges"][token] = cur
            p = s[p]["link"]
        if p != -1:
            q = s[p]["edges"][token]
            if s[p]["length"] + 1 == s[q]["length"]:
                s[cur]["link"] = q
            else:
                clone = len(s)
                s.append(dict(link = s[q]["link"], length = s[p]["length"] + 1,
                              end = s[q]["end"], edges = s[q]["edges"].copy()))
                while p != -1 and s[p]["edges"].get(token) == q:
                    s[p]["edges"][token] = clone
                    p = s[p]["link"]
                s[q]["link"] = s[cur]["link"] = clone
        self.last = cur
        self.pos += 1
        return result


def brute_accept(history, token):
    stream = history + [token]
    for length in range(len(history), 0, -1):
        suffix = stream[-length:]
        for start in range(len(history) - length + 1):
            if history[start:start + length] == suffix:
                return start, start + length
    return -1, -1


_rng = random.Random(17)
SMALL_SEQUENCES = {
    "run": [3] * 80,
    "period3": [1, 2, 3] * 35,
    "random": [_rng.randrange(9) for _ in range(250)],
}


@pytest.mark.parametrize("name", SMALL_SEQUENCES)
def test_small_sequences_against_suffix_search(name):
    sam, ref, history = ext.BC_SAM(), ReferenceSAM(), []
    for token in SMALL_SEQUENCES[name]:
        expected = brute_accept(history, token)
        assert ref.accept(token) == expected
        assert sam.accept(token) == expected
        history.append(token)


def test_promotions_clones_reset_and_sparse_ids():
    rng = random.Random(42)
    # Repeated hubs acquire >64 children. Prefix variants exercise cloning.
    tokens = []
    for _ in range(5):
        children = list(range(128))
        rng.shuffle(children)
        for t in children:
            tokens.extend([rng.randrange(4) + 500, 999, t, 777, t])
    special = [-2147483648, -1, 0, 255, 256, 1048575, 1048576, 2147483647]
    tokens += [rng.choice(special) for _ in range(2000)]
    tokens += [rng.randrange(300) for _ in range(20000)]
    sam = ext.BC_SAM()
    for reserve in (0, len(tokens)):
        sam.reset(reserve)
        ref = ReferenceSAM()
        for token in tokens:
            assert sam.accept(token) == ref.accept(token)
        assert sam.length() == len(tokens)
    sam.reset(0)
    assert sam.accept(999) == (-1, -1)
    assert sam.accept(-1) == (-1, -1)


def test_tensor_chunks_and_rewind():
    rng = random.Random(7)
    tokens = [x for i in range(180) for x in (700, i, 900, i)]
    tokens += [rng.randrange(200) for _ in range(1500)]
    tensor = torch.tensor(tokens, dtype = torch.long).view(1, -1)
    sam, ref, offset = ext.BC_SAM(), ReferenceSAM(), 0
    for end in list(range(1, len(tokens), 8)) + [len(tokens)]:
        for token in tokens[offset:end]:
            expected = ref.accept(token)
        assert sam.accept_tensor(tensor[:, :end]) == expected
        assert sam.length() == end
        assert sam.accept_tensor(tensor[:, :end]) == (-1, -1)
        offset = end
    for end in (900, 100, 0, 1000, len(tokens)):
        fresh = ext.BC_SAM()
        assert sam.accept_tensor(tensor[:, :end]) == fresh.accept_tensor(tensor[:, :end])
        assert sam.length() == end
