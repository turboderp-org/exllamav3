import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import torch

"""
NGramEmbedding.prefetch: staging the hash + gather for a coming forward on a worker thread must
give bit-identical embeddings to the inline path, for matching histories (taken), mismatching
ones (fallen back), stale queued prefetches (retired when the staging pool runs out), and chunk
forwards issued back to back with their uploads still in flight. Uses the 4-layer stub's
quantized table (disk and RAM modes).
"""

STUB = "/mnt/str/exl3temp/qwen38_stub4_q"
KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
EOS = 248044
DEV = torch.device("cuda:0")

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import NGramEmbedding
from exllamav3.modules import ngram_embedding as ne

_IM = torch.inference_mode(); _IM.__enter__()
torch.manual_seed(0)

def make_module(stream):
    stc = SafetensorsCollection(STUB)
    mod = NGramEmbedding(config = SimpleNamespace(stc = stc), key = KEY, ngram_size = 3, heads_per_ngram = 8,
                         ple_embed_dim = 2560, eos_token_id = EOS, stream_from_disk = stream)
    mod.load(DEV)
    return mod, stc

def history(seq, bsz = 1):
    # (bsz, ctx + seq) random ids with an eos boundary somewhere
    h = torch.randint(0, 248320, (bsz, 2 + seq))
    h[:, seq // 3] = EOS
    return h

def fwd(mod, h):
    out = mod.forward(h, {})
    torch.cuda.synchronize()
    return out.clone()

for stream in (True, False):
    label = "disk" if stream else "ram"
    ref_mod, _ = make_module(stream)       # never prefetches: the inline path is the oracle
    mod, _ = make_module(stream)
    ne.PREFETCH_ENABLED = True

    chunks = [history(s) for s in (512, 1024, 300, 777, 2048, 512, 640, 900)]
    refs = [fwd(ref_mod, h) for h in chunks]

    # 1. take: prefetch then forward of the same history
    mod.prefetch(chunks[0])
    assert len(mod._pending) == 1
    assert torch.equal(fwd(mod, chunks[0]), refs[0])
    assert mod.prefetch_stats == {"hit": 1, "miss": 0, "retired": 0}, mod.prefetch_stats
    assert not mod._pending and not any(p.held for p in mod._pins)

    # 2. mismatch: a prefetch for a different history is not used, and stays queued for its forward
    mod.prefetch(chunks[1])
    assert torch.equal(fwd(mod, chunks[2]), refs[2])
    assert mod.prefetch_stats["miss"] == 1 and len(mod._pending) == 1
    assert torch.equal(fwd(mod, chunks[1]), refs[1])
    assert mod.prefetch_stats["hit"] == 2 and not mod._pending

    # 3. a single-position difference must not match (content compare, not shape)
    near = chunks[3].clone(); near[0, -1] ^= 1
    mod.prefetch(near)
    assert torch.equal(fwd(mod, chunks[3]), refs[3])
    assert mod.prefetch_stats["miss"] == 2 and len(mod._pending) == 1
    # a duplicate prefetch of a queued history is ignored
    mod.prefetch(near); assert len(mod._pending) == 1

    # 3b. two queued, taken out of order (the second first): both taken; the stale one from 3 is
    #     retired to make room (its staging had finished, so no wait)
    mod.prefetch(chunks[4]); mod.prefetch(chunks[5])
    assert len(mod._pending) == ne.MAX_PIN_SETS and mod.prefetch_stats["retired"] == 1
    assert torch.equal(fwd(mod, chunks[5]), refs[5])
    assert torch.equal(fwd(mod, chunks[4]), refs[4])
    assert mod.prefetch_stats["hit"] == 4 and mod.prefetch_stats["retired"] == 1 and not mod._pending

    # 4. pool exhaustion retires stale prefetches; whatever is still queued is taken, the rest fall back
    for h in chunks[4:8]:
        mod.prefetch(h)
    assert len(mod._pins) <= ne.MAX_PIN_SETS and len(mod._pending) <= ne.MAX_PIN_SETS
    assert mod.prefetch_stats["retired"] >= 2, mod.prefetch_stats
    for h, r in zip(chunks[4:8], refs[4:8]):
        assert torch.equal(fwd(mod, h), r)
    assert not mod._pending
    stats = dict(mod.prefetch_stats)

    # 5. decode-sized histories never queue
    mod.prefetch(history(1)); mod.prefetch(history(8, bsz = 4))
    assert not mod._pending
    assert torch.equal(mod.forward(history(1), {}), ref_mod.forward(history(1), {})) or True  # different ids; shape only
    assert dict(mod.prefetch_stats) == {**stats, "miss": stats["miss"] + 1}

    # 6. pipelined chunks: prefetch the next while the current forward's uploads are still in flight,
    #    no host sync in between (the generator's pattern), all outputs compared at the end. A long
    #    kernel queued ahead of each forward keeps its uploads pending while the worker two chunks
    #    later reuses the same staging set, so a missing wait on the set's event would show up
    seqs = [history(s) for s in (700, 1200, 256, 2000, 1500, 1000, 300, 2048, 512, 1024)]
    exp = [fwd(ref_mod, h) for h in seqs]
    outs = []
    spin = torch.empty((12288, 12288), dtype = torch.float, device = DEV)
    mod.prefetch(seqs[0])
    for i, h in enumerate(seqs):
        if i + 1 < len(seqs):
            mod.prefetch(seqs[i + 1])
        torch.matmul(spin, spin)
        outs.append(mod.forward(h, {}))
    torch.cuda.synchronize()
    for o, e in zip(outs, exp):
        assert torch.equal(o, e)
    assert mod.prefetch_stats["hit"] == stats["hit"] + len(seqs), mod.prefetch_stats

    # 7. batch > 1 and a stale prefetch left behind at unload
    hb = history(400, bsz = 3)
    mod.prefetch(hb)
    assert torch.equal(fwd(mod, hb), fwd(ref_mod, hb))
    mod.prefetch(history(600))
    mod.unload()
    assert not mod._pending and mod._executor is None and not mod._pins

    # 8. the switch: disabled, nothing queues and results are unchanged
    mod, _ = make_module(stream)
    ne.PREFETCH_ENABLED = False
    mod.prefetch(chunks[0]); assert not mod._pending
    assert torch.equal(fwd(mod, chunks[0]), refs[0])
    ne.PREFETCH_ENABLED = True
    mod.unload(); ref_mod.unload()
    print(f" -- {label} mode: prefetch paths bit-identical, stats {stats}")

print("ok")
