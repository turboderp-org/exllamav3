"""A cut that does not fall on a pre-token boundary makes the chunked ids differ from the whole-text
ids, and a growing prompt then reuses none of its cached prefix. Which positions are boundaries belongs
to the tokenizer, so the cuts are asked of the tokenizer (issues #564 and #599 are one cut inside a
blank line and inside a whitespace-only line, respectively). The fixtures span five pre-tokenizer
shapes over a raw single-byte BPE: pieces starting at every whitespace run, a prefix space on
word-initial pieces, runs glued to the word before or after them, and a run whose ends glue apart at
its newlines, so the cut just past the newlines is the only boundary. The gluing shapes carry BPE
merges that fire only on the uncut fused piece, so a cut inside it is visible in the ids and must be
rejected (the cut falls to the surviving candidate of the group) or, when no candidate survives, drop
the whole group."""

import pytest
from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers

from exllamav3.tokenizer.tokenizer import (Tokenizer, _CHUNK_MIN_CHARS, _CHUNK_TARGET_CHARS, _cut_positions,
                                           _split_at)

pytestmark = pytest.mark.nogpu
SPECIAL = "<|control|>"
WHITESPACE = ["\n\n", "\n \n", "\n\t\n", "\r\n\r\n"]


class _FusingRun:
    """Byte-level-style pieces except that whitespace runs glue to the word on `side`: "previous"
    glues every run to the word before it, "next" to the word after it, "both" glues a run containing
    a newline to the word on both sides (the issue #599 shape, a whitespace-only line as one piece),
    and "newline-prev" splits a run at its newlines: the word before and the newlines are one piece,
    the trailing whitespace and the word after are the next (the shape that leaves the cut just past
    the newlines as the only boundary of the run). All of them leave the other runs whole. The merges
    of `_make_backend` fire only on an uncut fused piece, so a cut inside one is visible in the ids
    and must be rejected or dropped"""

    def __init__(self, side: str):
        self.side = side

    def pre_tokenize(self, seq):
        seq.split(self._split)

    def _split(self, index, s):
        out, t, i, n = [], s.normalized, 0, len(s.normalized)
        while i < n:
            j = i
            while j < n and not t[j].isspace():
                j += 1
            k = j
            while k < n and t[k].isspace():
                k += 1
            m = k
            while m < n and not t[m].isspace():
                m += 1
            if self.side == "next":
                if i < j:
                    out.append(s.slice((i, j)))
                if m > j:
                    out.append(s.slice((j, m)))
                i = m
            elif self.side == "previous":
                out.append(s.slice((i, k)))
                i = k
            elif self.side == "newline-prev":
                # a run with a newline splits where its newlines end: the word before and the newlines
                # are one piece, the trailing whitespace and the word after are the next; a run at the
                # start of the text glues forward, like the real tokenizers of this shape, so a cut
                # that splits the run does not orphan its trailing whitespace
                if i == j:
                    if m > j:
                        out.append(s.slice((j, m)))
                    else:
                        out.append(s.slice((j, k)))
                    i = m
                elif "\n" in t[j:k]:
                    q = t.find("\n", j)
                    while q < k and t[q] in "\r\n":
                        q += 1
                    if i < q:
                        out.append(s.slice((i, q)))
                    if m > q:
                        out.append(s.slice((q, m)))
                    i = m
                else:
                    if i < j:
                        out.append(s.slice((i, j)))
                    if j < k:
                        out.append(s.slice((j, k)))
                    i = k
            else:
                if "\n" in t[j:k]:
                    if m > k:
                        out.append(s.slice((i, m)))
                    else:
                        out.append(s.slice((i, k)))
                    i = m
                else:
                    if i < j:
                        out.append(s.slice((i, j)))
                    if j < k:
                        out.append(s.slice((j, k)))
                    i = k
        return out


def _make_backend(param: str) -> HFTokenizer:
    """A single-byte BPE backend with the pre-tokenizer of `param`. The built-in ByteLevel pieces
    are byte-level escaped, the fused pieces are not, so each family takes the matching vocab. The
    word before and after the runs of the test texts is "text", so the merges straddle exactly the
    fused pieces the runs glue into; the special token gets an id above the merge tokens"""
    if param in ("byte-level", "prefix-space"):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {c: i for i, c in enumerate(alphabet)}
        merges = []
        pretok = pre_tokenizers.ByteLevel(add_prefix_space = param == "prefix-space")
    else:
        vocab = {chr(i): i for i in range(256)}
        merges = []
        side = "both" if param == "fuses-both" else param.split("-", 1)[1]
        # "both" also carries the "\n " merge, so a cut at any of the three candidates of a " \n \n"
        # run straddles a merge and the whole group drops
        next_id = 256
        if side in ("previous", "newline-prev") or side == "both":
            vocab["t "] = next_id
            merges.append(("t", " "))
            next_id += 1
        if side == "both":
            vocab["\n "] = next_id
            merges.append(("\n", " "))
            next_id += 1
        if "next" in side or side == "both":
            vocab["\nt"] = next_id
            merges.append(("\n", "t"))
            next_id += 1
        if side == "newline-prev":
            vocab[" d"] = next_id
            merges.append((" ", "d"))
            next_id += 1
        vocab[SPECIAL] = next_id
        pretok = pre_tokenizers.PreTokenizer.custom(_FusingRun(side))
    backend = HFTokenizer(models.BPE(vocab = vocab, merges = merges))
    backend.pre_tokenizer = pretok
    backend.add_special_tokens([SPECIAL])
    return backend

def _make_tok(backend: HFTokenizer) -> Tokenizer:
    tok = Tokenizer.__new__(Tokenizer)
    tok.tokenizer = backend
    tok.bos_token_id = tok.eos_token_id = None
    tok.pad_token_id = -1
    tok.unspecial_piece_to_id = {}
    tok.missing_special_piece_to_id = {}
    return tok


@pytest.fixture(params = ["byte-level", "prefix-space", "fuses-previous", "fuses-next"])
def tokenizer(request):
    """One pre-tokenizer per shape of whitespace handling: pieces starting at every whitespace run,
    a prefix space on word-initial pieces (so a cut after the run sits before a prefixed word), and
    runs glued to the word before or after them, the issue #564 and #599 shapes. Each of these cuts
    the shared text, at the run start or the run end"""
    return _make_tok(_make_backend(request.param))


def encode_chunks(backend, chunks):
    return [i for c in chunks for i in backend.encode(c, add_special_tokens = False).ids]


@pytest.mark.parametrize("whitespace", WHITESPACE)
def test_chunks_encode_as_the_whole_text(tokenizer, whitespace):
    """However a long prompt gets cut, its chunks must encode to the whole text's ids: that is all the
    prefix cache asks of the splitter"""
    text = (f"text {SPECIAL} more text" + " " + whitespace) * (_CHUNK_MIN_CHARS // 20)
    backend = tokenizer.tokenizer
    assert encode_chunks(backend, tokenizer._chunks(text)) == backend.encode(
        text, add_special_tokens = False).ids


@pytest.mark.parametrize("whitespace", WHITESPACE)
def test_growing_prompt_keeps_its_prefix(tokenizer, whitespace):
    """Appending to a prompt must not change the ids of what was already there: where the cuts fall
    depends on the length of the text, which is how a cut inside a token showed up"""
    text = (f"text {SPECIAL} more text" + " " + whitespace) * (_CHUNK_MIN_CHARS // 20)
    backend = tokenizer.tokenizer
    for growed in (text, text + " and then a bit of text", text + "\n \n"):
        assert encode_chunks(backend, tokenizer._chunks(growed)) == backend.encode(
            growed, add_special_tokens = False).ids


@pytest.mark.parametrize("whitespace", WHITESPACE)
def test_cuts_are_made_and_never_split_a_token(tokenizer, whitespace):
    """Candidates get checked and then cut, so a chunk is never empty nor the whole text, and the
    chunks still spell out the text they were cut from"""
    text = (f"text {SPECIAL} more text" + " " + whitespace) * (_CHUNK_MIN_CHARS // 10)
    chunks = tokenizer._chunks(text)
    assert all(chunks) and "".join(chunks) == text
    assert len(chunks) > 1


@pytest.mark.parametrize("special", [False, True])
@pytest.mark.parametrize("whitespace", WHITESPACE)
def test_long_encode_preserves_special_token_mode(tokenizer, special, whitespace):
    """The boundary checks must not change the special-token mode of the actual encode"""
    line = f"text {SPECIAL} text" + whitespace
    text = line * (_CHUNK_MIN_CHARS // len(line) + 1)
    tokenizer.tokenizer.encode_special_tokens = not special
    expected = tokenizer.tokenizer.encode(text, add_special_tokens = False).ids
    control_id = tokenizer.tokenizer.token_to_id(SPECIAL)
    assert (control_id in expected) == special

    first = tokenizer.encode(text, encode_special_tokens = special)[0].tolist()
    second = tokenizer.encode(text, encode_special_tokens = special)[0].tolist()

    assert first == second == expected
    assert tokenizer.tokenizer.encode_special_tokens == (not special)


# The run " \n \n" of this text starts at 8, its newlines end at 10, and it ends at 12, where the
# second "text" begins
CUT_TEXT = "aaa text" + " \n \n" + "text ccc"
RUN_START, RUN_NEWS, RUN_END = 8, 10, 12


@pytest.mark.parametrize(
    "param, from_both, from_run_end",
    [("byte-level", [RUN_NEWS], [RUN_END]),
     ("prefix-space", [RUN_NEWS], []),
     ("fuses-previous", [RUN_NEWS], [RUN_END]),
     ("fuses-next", [RUN_NEWS], []),
     ("fuses-both", [], []),
     ("fuses-newline-prev", [RUN_NEWS], [RUN_END])])
def test_candidate_selection(param, from_both, from_run_end):
    """Which candidate the check takes, over a single group, in preference order (the cut just past the
    newlines, then the run start, then the run end): the mid cut wherever the run is its own piece or
    the newlines glue backward, the run start when the mid cut sits inside the fused run+word piece,
    and no cut at all when the run glues to the word on both sides, so no candidate lands on a
    boundary"""
    tok = _make_tok(_make_backend(param))
    assert tok._verified_cuts(CUT_TEXT, [[RUN_NEWS, RUN_START, RUN_END]]) == from_both
    assert tok._verified_cuts(CUT_TEXT, [[RUN_END]]) == from_run_end


@pytest.mark.parametrize(
    "param, check_batches",
    [("byte-level", [3]),
     ("prefix-space", [3]),
     ("fuses-previous", [3]),
     ("fuses-next", [3]),
     ("fuses-both", [3, 6]),
     ("fuses-newline-prev", [3])])
def test_check_batches_the_first_candidate_alone(param, check_batches, monkeypatch):
    """The mid cut is windowed in one batch; only the other candidates of the groups that fail it
    (the run start and the run end, six windows) take a second batch, so a chunkable tokenizer pays
    one check batch per encode"""
    tok = _make_tok(_make_backend(param))
    calls = []
    encode_batch = tok.tokenizer.encode_batch

    def counting(batch, *args, **kwargs):
        calls.append(len(batch))
        return encode_batch(batch, *args, **kwargs)

    monkeypatch.setattr(tok.tokenizer, "encode_batch", counting)
    tok._verified_cuts(CUT_TEXT, [[RUN_NEWS, RUN_START, RUN_END]])
    assert calls == check_batches


# The run " \n\n    " of this text starts at 8, its newlines end at 11, and it ends at 15, where the
# fused "    def" piece continues
MID_TEXT = "aaa text \n\n    def ccc"
MID_START, MID_NEWS, MID_END = 8, 11, 15


def test_mid_cut_selected_when_the_run_ends_glue_apart():
    """Where the run start sits inside the fused word+newlines piece and the run end inside the fused
    spaces+word piece, the only boundary of the run is just past its newlines -- the cut the old
    splitter always used, which the run-start/run-end pair misses -- and the second batch finds it"""
    tok = _make_tok(_make_backend("fuses-newline-prev"))
    assert tok._verified_cuts(MID_TEXT, [[MID_START, MID_NEWS, MID_END]]) == [MID_NEWS]
    assert tok._verified_cuts(MID_TEXT, [[MID_NEWS]]) == [MID_NEWS]
    assert tok._verified_cuts(MID_TEXT, [[MID_START]]) == []
    assert tok._verified_cuts(MID_TEXT, [[MID_END]]) == []


def test_no_surviving_cut_encodes_whole():
    """A prompt none of whose cut candidates survive encodes whole, single-threaded, and correct"""
    tok = _make_tok(_make_backend("fuses-both"))
    line = f"text {SPECIAL} more text" + " \n \n"
    text = line * (3 * _CHUNK_TARGET_CHARS // len(line) + 1)
    groups = _cut_positions(text)
    assert len(groups) >= 2
    assert tok._verified_cuts(text, groups) == []
    assert tok._chunks(text) == [text]
    assert encode_chunks(tok.tokenizer, [text]) == tok.tokenizer.encode(
        text, add_special_tokens = False).ids


def test_partial_survival_merges_only_the_failed_groups():
    """When some groups fail and some survive, the surviving cuts stand and the text merges across
    exactly the failed groups: with `fuses-next`, the run at the start of the text glues to the word
    after it, so its only candidate (the run end, the start being position 0) fails, while the next
    run's start sits at the start of its fused piece and survives"""
    tok = _make_tok(_make_backend("fuses-next"))
    text = " \n \ntext" + "  \n \n" + "more text"
    # the first run is 0..4, fused with "text" into one piece; the second is 8..13, fused with "more text"
    cuts = tok._verified_cuts(text, [[4], [8, 13]])
    assert cuts == [8]
    chunks = _split_at(text, cuts)
    assert chunks == [" \n \ntext", "  \n \nmore text"]
    assert encode_chunks(tok.tokenizer, chunks) == tok.tokenizer.encode(
        text, add_special_tokens = False).ids


def test_trailing_run_reaches_the_end():
    """A run that ends at the text's end offers only its start as a candidate (the end is not a cut
    position), and once the cut lands there, the rest of the text is one whitespace run with nothing
    left to cut: _cut_positions stops at the empty group and the cut still holds"""
    wordy = ("w " * 12000).rstrip()
    text = wordy + "\n" * 26000
    a = len(wordy)
    assert _cut_positions(text) == [[a]]
    tok = _make_tok(_make_backend("byte-level"))
    assert tok._verified_cuts(text, [[a]]) == [a]
    assert encode_chunks(tok.tokenizer, tok._chunks(text)) == tok.tokenizer.encode(
        text, add_special_tokens = False).ids


def _make_unigram_backend() -> HFTokenizer:
    """A single-byte Unigram backend with the fusing pre-tokenizer of the `fuses-previous` shape:
    the fused piece "t " is what Viterbi picks on an uncut piece (its score beats the sum of the two
    bytes' scores; Viterbi here compares scores additively), so a cut inside it changes the ids,
    like the BPE merges of `_make_backend` do for the BPE shapes"""
    vocab = [(chr(i), 0.01) for i in range(256)]
    vocab += [("t ", 0.5), (SPECIAL, 0.99)]
    backend = HFTokenizer(models.Unigram(vocab = vocab, unk_id = 255))
    backend.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(_FusingRun("previous"))
    backend.add_special_tokens([SPECIAL])
    return backend


def test_unigram_backend_rejects_the_fused_cut():
    """The window check is a proof for BPE but a heuristic for non-BPE backends, whose Viterbi spans
    the whole pre-token: on a Unigram whose fused piece fires only on the uncut piece, the fused cut
    is still rejected and the boundary cut stands, and a long prompt still encodes as the whole text"""
    tok = _make_tok(_make_unigram_backend())
    assert tok._verified_cuts(CUT_TEXT, [[RUN_START, RUN_END]]) == [RUN_END]
    text = (f"text {SPECIAL} more text" + " " + "\n \n") * (_CHUNK_MIN_CHARS // 20)
    assert encode_chunks(tok.tokenizer, tok._chunks(text)) == tok.tokenizer.encode(
        text, add_special_tokens = False).ids
