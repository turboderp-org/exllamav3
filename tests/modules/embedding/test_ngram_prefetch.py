"""
NGramEmbedding.prefetch: staging the hash + gather for a coming forward on a worker thread must give bit-identical
embeddings to the inline path (a second module instance that never prefetches is the oracle), for matching histories
(taken), mismatching ones (fallen back), stale queued prefetches (retired when the staging pool runs out), and chunk
forwards issued back to back with their uploads still in flight. Uses the quantized n-gram table of the
qwen4-exp-stub registry model, in disk-streamed and RAM modes.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import NGramEmbedding
from exllamav3.modules import row_table as rt

KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
EOS = 248044
VOCAB = 248320

pytestmark = pytest.mark.model("qwen4-exp-stub")


@pytest.fixture(scope = "module", autouse = True)
def _inference_mode():
    # Entered per module, not at import: an import-time __enter__ leaked inference mode into every test module
    # collected after this one
    with torch.inference_mode():
        yield


@pytest.fixture
def make_module(model_dir, device):
    made = []

    def make(stream):
        mod = NGramEmbedding(config = SimpleNamespace(stc = SafetensorsCollection(model_dir)), key = KEY,
                             ngram_size = 3, heads_per_ngram = 8, ple_embed_dim = 2560, eos_token_id = EOS,
                             stream_from_disk = stream)
        mod.load(device)
        made.append(mod)
        return mod

    yield make
    for mod in made:
        mod.unload()


def history(seq, bsz = 1):
    """(bsz, ctx + seq) random ids with an eos boundary somewhere"""
    h = torch.randint(0, VOCAB, (bsz, 2 + seq))
    h[:, seq // 3] = EOS
    return h


def fwd(mod, h):
    out = mod.forward(h, {})
    torch.cuda.synchronize(mod.device)
    return out.clone()


@pytest.mark.parametrize("stream", [True, False], ids = ["disk", "ram"])
def test_prefetch_paths(make_module, device, monkeypatch, stream):
    torch.manual_seed(0)
    ref_mod = make_module(stream)       # never prefetches: the inline path is the oracle
    mod = make_module(stream)
    monkeypatch.setattr(rt, "PREFETCH_ENABLED", True)

    chunks = [history(s) for s in (512, 1024, 300, 777, 2048, 512, 640, 900)]
    refs = [fwd(ref_mod, h) for h in chunks]

    # 1. take: prefetch then forward of the same history
    mod.prefetch(chunks[0])
    assert len(mod.table._pending) == 1
    assert torch.equal(fwd(mod, chunks[0]), refs[0])
    assert mod.prefetch_stats == {"hit": 1, "miss": 0, "retired": 0}, mod.prefetch_stats
    assert not mod.table._pending and not any(p.held for p in mod.table._pins)

    # 2. mismatch: a prefetch for a different history is not used, and stays queued for its forward
    mod.prefetch(chunks[1])
    assert torch.equal(fwd(mod, chunks[2]), refs[2])
    assert mod.prefetch_stats["miss"] == 1 and len(mod.table._pending) == 1
    assert torch.equal(fwd(mod, chunks[1]), refs[1])
    assert mod.prefetch_stats["hit"] == 2 and not mod.table._pending

    # 3. a single-position difference must not match (content compare, not shape)
    near = chunks[3].clone()
    near[0, -1] ^= 1
    mod.prefetch(near)
    assert torch.equal(fwd(mod, chunks[3]), refs[3])
    assert mod.prefetch_stats["miss"] == 2 and len(mod.table._pending) == 1
    # a duplicate prefetch of a queued history is ignored
    mod.prefetch(near)
    assert len(mod.table._pending) == 1

    # 3b. two queued, taken out of order (the second first): both taken; the stale one from 3 is retired to make
    #     room (its staging had finished, so no wait)
    mod.prefetch(chunks[4])
    mod.prefetch(chunks[5])
    assert len(mod.table._pending) == rt.MAX_PIN_SETS and mod.prefetch_stats["retired"] == 1
    assert torch.equal(fwd(mod, chunks[5]), refs[5])
    assert torch.equal(fwd(mod, chunks[4]), refs[4])
    assert mod.prefetch_stats["hit"] == 4 and mod.prefetch_stats["retired"] == 1 and not mod.table._pending

    # 4. pool exhaustion retires stale prefetches; whatever is still queued is taken, the rest fall back
    for h in chunks[4:8]:
        mod.prefetch(h)
    assert len(mod.table._pins) <= rt.MAX_PIN_SETS and len(mod.table._pending) <= rt.MAX_PIN_SETS
    assert mod.prefetch_stats["retired"] >= 2, mod.prefetch_stats
    for h, r in zip(chunks[4:8], refs[4:8]):
        assert torch.equal(fwd(mod, h), r)
    assert not mod.table._pending
    stats = dict(mod.prefetch_stats)

    # 5. decode-sized histories never queue, and their forward (one inline lookup, a miss) matches the oracle
    mod.prefetch(history(1))
    mod.prefetch(history(8, bsz = 4))
    assert not mod.table._pending
    h1 = history(1)
    assert torch.equal(fwd(mod, h1), fwd(ref_mod, h1))
    assert dict(mod.prefetch_stats) == {**stats, "miss": stats["miss"] + 1}
    stats = dict(mod.prefetch_stats)

    # 6. pipelined chunks: prefetch the next while the current forward's uploads are still in flight, no host sync
    #    in between (the generator's pattern), all outputs compared at the end. A long kernel queued ahead of each
    #    forward keeps its uploads pending while the worker two chunks later reuses the same staging set, so a
    #    missing wait on the set's event would show up
    seqs = [history(s) for s in (700, 1200, 256, 2000, 1500, 1000, 300, 2048, 512, 1024)]
    exp = [fwd(ref_mod, h) for h in seqs]
    outs = []
    spin = torch.empty((12288, 12288), dtype = torch.float, device = device)
    mod.prefetch(seqs[0])
    for i, h in enumerate(seqs):
        if i + 1 < len(seqs):
            mod.prefetch(seqs[i + 1])
        torch.matmul(spin, spin)
        outs.append(mod.forward(h, {}))
    torch.cuda.synchronize(device)
    for o, e in zip(outs, exp):
        assert torch.equal(o, e)
    assert mod.prefetch_stats["hit"] == stats["hit"] + len(seqs), mod.prefetch_stats
    del spin

    # 7. batch > 1 and a stale prefetch left behind at unload
    hb = history(400, bsz = 3)
    mod.prefetch(hb)
    assert torch.equal(fwd(mod, hb), fwd(ref_mod, hb))
    mod.prefetch(history(600))
    table = mod.table
    mod.unload()
    assert not table._pending and table._executor is None and not table._pins


@pytest.mark.parametrize("stream", [True, False], ids = ["disk", "ram"])
def test_prefetch_disabled(make_module, monkeypatch, stream):
    """The switch: disabled, nothing queues and results are unchanged"""
    torch.manual_seed(1)
    ref_mod = make_module(stream)
    mod = make_module(stream)
    monkeypatch.setattr(rt, "PREFETCH_ENABLED", False)
    h = history(512)
    mod.prefetch(h)
    assert not mod.table._pending
    assert torch.equal(fwd(mod, h), fwd(ref_mod, h))
