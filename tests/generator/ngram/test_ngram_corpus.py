"""
Frozen n-gram corpus (NgramCorpus) loaded from a bank written by util/build_sam.py: cursor matches equal the
longest history suffix found in the corpus by brute-force search, drafts continue an actual occurrence, cursors
are independent and outlive the bank, live-SAM vs corpus draft selection in Job.get_ngram_draft, and rejection
of mismatched tokenizers, inconsistent graphs, corrupt frames and truncated files.
"""

import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exllamav3.generator.job import Job
from exllamav3.generator.ngram import NgramCorpus
from testlib.repo_scripts import load_repo_script

zstandard = pytest.importorskip("zstandard")

pytestmark = pytest.mark.nogpu


class Bank:
    """A small corpus bank in a temporary model directory, with a stub tokenizer pointing at it"""

    def __init__(self, path):
        from exllamav3.ext import exllamav3_ext as ext
        self.builder = load_repo_script("util/build_sam.py")
        self.path = path
        (path / "tokenizer.json").write_text("fixture")
        self.tokenizer = SimpleNamespace(config = SimpleNamespace(directory = path), actual_vocab_size = 1000)
        rng = random.Random(17)
        self.ids = [1, 2, 3, 4, -1, 5, 6, 7, -1]
        self.ids += [rng.randrange(20) for _ in range(300)] + [-1]
        sam = ext.BC_SAM()
        sam.accept_tensor(torch.tensor(self.ids))
        self.arrays = dict(zip(self.builder.GRAPH_ARRAYS, [a.numpy() for a in sam.export_csr()]))
        self.arrays["corpus"] = np.array(self.ids, dtype = "<i4")
        self.arrays["document_ends"] = np.flatnonzero(self.arrays["corpus"] == -1).astype("<i4")
        self.meta = dict(
            tokenizer = {"files": {"tokenizer.json": hashlib.sha256(b"fixture").hexdigest()}},
            state_count = len(self.arrays["link"]),
            edge_count = len(self.arrays["edge_to"]),
            token_count = len(self.ids),
            document_count = 3,
            corpus_sha256 = hashlib.sha256(self.arrays["corpus"].tobytes()).hexdigest(),
        )
        self.file = path / "test.sam.zst"
        self.save()

    def save(self):
        self.builder.write_bank(self.file, self.arrays, self.meta, force = True)

    def occurs(self, seq):
        n = len(seq)
        return any(self.ids[i:i + n] == seq for i in range(len(self.ids) - n + 1))


@pytest.fixture
def bank(tmp_path):
    return Bank(tmp_path)


def test_cursor_matches_rewinds_and_independence(bank):
    corpus = NgramCorpus(bank.file, bank.tokenizer)
    cursor, other = corpus.cursor(), corpus.cursor()
    history = []
    rng = random.Random(42)
    for step in range(300):
        if step % 37 == 0:
            history = []
            cursor.draft(torch.empty((1, 0), dtype = torch.long), 1, 15)
        history.extend([rng.randrange(22) for _ in range(rng.randrange(1, 5))])
        matched, draft = cursor.draft(torch.tensor([history]), 1, 15)
        expected = 0
        for length in range(1, len(history) + 1):
            if bank.occurs(history[-length:]):
                expected = length
        assert matched == expected
        assert -1 not in draft.tolist()[0]
        if draft.numel():
            assert bank.occurs(history[-matched:] + draft.tolist()[0])
    match, draft = other.draft(torch.tensor([[1, 2, 3]]), 2, 15)
    assert (match, draft.tolist()) == (3, [[4]])
    assert other.draft(torch.tensor([[1, 2, 3, 4]]), 2, 15)[1].numel() == 0
    # Cursor references keep the payload alive independently of the bank.
    del corpus
    assert other.draft(torch.tensor([[5, 6]]), 2, 15)[1].tolist() == [[7]]


def test_loader_rejects_bad_graph_and_tokenizer(bank):
    (bank.path / "tokenizer.json").write_text("different")
    with pytest.raises(ValueError, match = "tokenizer mismatch"):
        NgramCorpus(bank.file, bank.tokenizer)
    (bank.path / "tokenizer.json").write_text("fixture")
    bank.arrays["link"][1] = 1
    bank.save()
    with pytest.raises(ValueError, match = "graph"):
        NgramCorpus(bank.file, bank.tokenizer)


def test_live_vs_corpus_selection(bank):
    from exllamav3.ext import exllamav3_ext as ext
    corpus = NgramCorpus(bank.file, bank.tokenizer)
    seq = torch.tensor([[1, 2, 3]])
    job = SimpleNamespace(
        sam = ext.BC_SAM(),
        corpus_cursor = corpus.cursor(),
        generator = SimpleNamespace(ngram_match_min = 2),
        sequences = [SimpleNamespace(sequence_ids = SimpleNamespace(torch = lambda: seq))],
    )
    assert Job.get_ngram_draft(job, 15).tolist() == [[4]]
    # Live tie wins, even if the corpus has a different continuation.
    seq = torch.tensor([[1, 2, 9, 1, 2]])
    job.sam = ext.BC_SAM()
    job.corpus_cursor = corpus.cursor()
    assert Job.get_ngram_draft(job, 1).tolist() == [[9]]


def test_corrupt_frame(bank):
    original = bank.file.read_bytes()
    damaged = bytearray(original)
    damaged[-1] ^= 1
    bank.file.write_bytes(damaged)
    with pytest.raises(zstandard.ZstdError):
        NgramCorpus(bank.file, bank.tokenizer)
    bank.file.write_bytes(original[:10])
    with pytest.raises(ValueError, match = "Truncated"):
        NgramCorpus(bank.file, bank.tokenizer)
