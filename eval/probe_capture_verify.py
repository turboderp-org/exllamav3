"""Verify-forward capture probe (feasibility gate for CUDA graphs).

Rescoped per review: capture the TARGET VERIFY forward only (the 914-launch
cluster from target_ab.py), never the draft->selector->verify round. Reject
break, propose syncs, and the accept loop stay on the host.

Gate checklist (all verified by reading before running):
- cache_seqlens: device tensor, tl.load in Triton path (torch/xformers .item()
  fallbacks are not taken when attn_mode=flash_attn).
- block_table: shape-key cached, fixed at batch-1/fixed-window; statics+copy_.
- Warmup on a SIDE stream (default-stream warmup after live allocations is the
  documented silent-poison mode).

Run ONLY when :8290 is down (second 27B does not fit beside serve), or adapt
to a single-TransformerBlock micro-forward (weaker gate: a pass does not prove
48-layer capture, but a fail is a real blocker).

Success: 20 replays, argmax token-id match vs eager every replay, dispatch
gap shrink visible on the verify slice. Any divergence or capture throw names
the first EXL3 op to replace -- that op is the kernel task.
"""

import torch


def capture_verify(model, cache, ids, seqlens, block_table, positions, n_check=20):
    dev = torch.device("cuda")
    def static(t):
        # Capture poisons on any CPU->CUDA copy inside the region
        # (prepare_for_device/to_device). Everything static lives on CUDA;
        # replay mutates via device-to-device copy_ only.
        return t.to(dev).contiguous() if torch.is_tensor(t) else t
    static = {
        "ids": static(ids),
        "seqlens": static(seqlens),
        "bt": static(block_table),
        "pos": static(positions),
    }
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "cache_seqlens": static["seqlens"],
        "block_table": static["bt"],
        "positions": static["pos"],
    }

    # 1. warm up on a SIDE stream (allocations, autotune, JIT stay out of capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model.forward(static["ids"], params)
    torch.cuda.current_stream().wait_stream(s)

    # 2. capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = model.forward(static["ids"], params)

    # 3. replay protocol: copy_ into statics, never reassign, never re-capture.
    # static_out is a REUSED buffer -- clone before the next replay.
    for r in range(n_check):
        static["ids"].copy_(ids)
        static["seqlens"].copy_(seqlens)
        static["bt"].copy_(block_table)
        static["pos"].copy_(positions)
        g.replay()
        got = static_out.clone()
        want = model.forward(static["ids"], params)
        if not torch.equal(got.argmax(-1), want.argmax(-1)):
            print(f"DIVERGE at replay {r}")
            return False
    print(f"probe PASS: {n_check} replays, argmax match")
    return True


if __name__ == "__main__":
    raise SystemExit(
        "wire-up script: load target+cache via model_init (serve DOWN), "
        "build fixed window-8 ids/seqlens/block_table/positions, call capture_verify"
    )
