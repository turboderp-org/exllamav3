"""
Synthetic latency sweep of ONE DSV4 attention block in isolation (BC/graph decode paths
enabled), over bsz 1..4 x seqlen 1..8 at a fixed past length. Sanity check for the
attention block's latency scaling across the two batch dimensions, isolated from the MoE
layers, and a fixed reference point for measuring optimization progress.

Method:
  - Loads the full model, picks the first attention layer of the requested type, and
    drives ONLY that module: per-slot test states at --past, module.forward() in a loop.
    bsz 1 goes through the whole-step BC graph (EXL3_BC_DSA), bsz > 1 through the batched
    (B, S) graph path.
  - Pools, rings and compressor buffers are randomized so indexer scores are spread and
    the top-k gather hits scattered pool entries (zeros would select a contiguous block
    and understate gather cost).
  - Timing is CUDA-event based, two numbers per config:
      warm = events around each step, back-to-back loop (pipelined, caches warm)
      cold = a 256 MB L2-thrashing sweep between launches, events bracketing only the
             step: single-launch latency with all weights/state streamed from DRAM
  - GB/s and GFLOP/s columns divide an ANALYTIC traffic/FLOP model by the cold latency:
      bytes(B,S) = W_q (quantized weights, streamed once per forward, shared by all rows)
                 + B*S * [ indexer key-pool scan (ec * D_i * 2)
                         + scores write + topk read (~3 * ec * 2)
                         + split-kernel K=V reads once per head block ((win+k) * D * 2 * hb)
                         + split/combine fp32 workspace round trip (2 * H * splits * D * 4) ]
      flops(B,S) = B*S * [ 2 * proj_params + 4 * H * D * (win+k) + 2 * H_i * D_i * ec ]

Reading the results:
  - Weight streaming dominates until B*S ~ 7; per-row state takes over past that.
  - The bandwidth-saturated corner (high B*S) should sit near the device's achievable
    DRAM bandwidth; the low-R corner is limited by the serial small-kernel chain inside
    the graph, not bandwidth.
  - cold - warm ~ the L2-resident fraction of the weight stream being re-fetched.
  - A latency step where R = B*S crosses 16 is the exl3 mgemm multi-row kernel tier.
  - An isolated blown cold cell = a Triton JIT compile landing in the timed loop (a ring
    shift makes the BC path decline to the eager core for one step at a shape whose eager
    kernels weren't compiled yet). Rerun that cell, don't chase it.

Reference (2026-08-04, dev_dsv4 paged pools + C++ BC batched graphs + AOT divisibility
attrs + M >= 8 padded wts hgemm, V4-Flash 3.00bpw, CSA layer 2 on the 4090, past 65536,
ec 16384): weights 79.4 MB / 11.11 MB/row / 605 MFLOP/row;
  B1 S1: warm 234us | B2 S1: 241 | B4 S1: 258 | B2 S8: 325 | B4 S4: 325
  B4 S8: warm 516 cold 522 (833 GB/s, 37 TFLOP/s)
The grid should be MONOTONIC in both B and S (modulo the R > 16 mgemm tier step). A ~60us
bump in any R = 2..7 cell means the wts-GEMM M padding regressed: cuBLASLt's heuristic for
the (R x 4096) @ (4096 x 64) fp16 GEMM picks a ~63us kernel for M = 2..7 vs ~5us at M >= 8,
which is why the hgemm runs over >= 8 zero-padded rows.
Full-model decode at these numbers: 85.1 / 194.1 / 256.2 tok/s at bsz 1 / 4 / 8 (32k cache,
short prompts, bench via jobs-tmp bench_batch.py; short context understates the padded-GEMM
gain, which only applies in the top-k regime).

Usage:
  python tests/bench_dsv4_attn_sweep.py -m /mnt/str/models/deepseek-v4-flash-0731/exl3/3.00bpw/
  [--past 65536] [--layer_type csa] [--max_bsz 4] [--max_seq 8]
"""
import argparse, time, torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache
from exllamav3.modules.dsv4 import DSV4Attention

ITERS_WARMUP = 24
ITERS_WARM = 64
ITERS_COLD = 32

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", default = "/mnt/str/models/deepseek-v4-flash-0731/exl3/3.00bpw/")
    p.add_argument("--past", type = int, default = 65536)
    p.add_argument("--layer_type", default = "csa", choices = ["csa", "hca", "sliding"])
    p.add_argument("--max_bsz", type = int, default = 4)
    p.add_argument("--max_seq", type = int, default = 8)
    args = p.parse_args()
    PAST = args.past

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    # Direct-forward (no page table) uses the slot-partitioned fallback: per-job pool
    # capacity = max_num_tokens / num_slots, so size the cache for max_bsz full-length jobs
    per_job = -(-(PAST + 2048) // 256) * 256
    cache = Cache(model, max_num_tokens = args.max_bsz * per_job, max_batch_size = args.max_bsz)
    model.load()

    mod = next(m for m in model if isinstance(m, DSV4Attention) and m.layer_type == args.layer_type)
    device = torch.device(mod.device) if not isinstance(mod.device, torch.device) else mod.device
    li = (mod.layer_idx, 0)
    kl = cache.layers.get(li)
    rsl = cache.get_recurrent_layer(li)
    print(f"layer {mod.layer_idx} ({mod.layer_type}) on {device}" +
          (f", epp {kl.epp}, pool capacity/job {kl.num_pages // args.max_bsz * kl.epp} entries"
           if kl else ""))

    torch.manual_seed(7)
    if kl:
        with torch.inference_mode():
            for t in kl.get_tensors():
                t.normal_(0, 0.5)

    # --- analytic model constants -----------------------------------------------------
    H, D = mod.num_q_heads, mod.head_dim
    win = mod.sliding_window
    has_idx = mod.indexer is not None
    Hi = mod.index_n_heads if has_idx else 0
    Di = mod.index_head_dim if has_idx else 0
    topk = mod.index_topk if has_idx else 0
    m_rate = mod.compress_rate if mod.compressor is not None else 1
    ec = PAST // m_rate if mod.compressor is not None else 0
    lins = [mod.q_a, mod.q_b, mod.wkv, mod.wo_b] + list(mod.wo_a) + [mod.idx_weights]
    if mod.compressor is not None:
        lins += [mod.compressor.wkv, mod.compressor.wgate]
    if has_idx:
        lins += [mod.idx_wq_b, mod.indexer.wkv, mod.indexer.wgate]
    w_bytes = sum(l.storage_size() for l in lins if l is not None)
    proj_params = sum(l.in_features * l.out_features_unpadded for l in lins if l is not None)

    hb = -(-H // 16)                      # head blocks (BLOCK_H 16)
    splits = 16
    keys = win + (topk if has_idx else min(ec, PAST))   # hca: dense pool scan
    row_bytes = (
        ec * Di * 2                       # indexer key-pool scan (csa)
        + (3 * ec * 2 if has_idx else 0)  # scores write + topk read
        + keys * D * 2 * hb               # split kernel K=V reads, once per head block
        + 2 * H * splits * D * 4          # ws_acc write + combine read (fp32)
    )
    row_flops = 2 * proj_params + 4 * H * D * keys + 2 * Hi * Di * ec
    print(f"weights {w_bytes / 1e6:.1f} MB quantized ({proj_params / 1e6:.0f} M params); "
          f"per-row state {row_bytes / 1e6:.2f} MB, per-row {row_flops / 1e6:.0f} MFLOP "
          f"(indexer {2 * Hi * Di * ec / 1e6:.0f} / proj {2 * proj_params / 1e6:.0f} "
          f"/ attn {4 * H * D * keys / 1e6:.0f})")

    thrash = torch.empty(64 * 1024 * 1024, dtype = torch.float, device = device)

    print(f"\n{'B':>2} {'S':>2} | {'warm us':>8} {'cold us':>8} {'tok/s(c)':>9} | "
          f"{'model GB':>9} {'-> GB/s':>8} | {'MFLOP':>8} {'-> GFLOP/s':>10}")

    for B in range(1, args.max_bsz + 1):
        for S in range(1, args.max_seq + 1):
            with torch.inference_mode():
                states = [cache.get_test_state(PAST) for _ in range(B)]
                for t in rsl._tensors():
                    t.normal_(0, 0.5)
            params = {"attn_mode": "flash_attn", "recurrent_states": states}
            x = torch.randn(B, S, mod.hidden_size, dtype = torch.half, device = device)

            def step():
                with torch.inference_mode():
                    y = mod.forward(x, dict(params))
                for rs in states:
                    rs.position += S
                    rs.post_advance()
                return y

            with torch.cuda.device(device):
                for _ in range(ITERS_WARMUP):
                    step()
                torch.cuda.synchronize(device)

                evs = [(torch.cuda.Event(enable_timing = True),
                        torch.cuda.Event(enable_timing = True)) for _ in range(ITERS_WARM)]
                for e0, e1 in evs:
                    e0.record()
                    step()
                    e1.record()
                torch.cuda.synchronize(device)
                lat_w = sum(e0.elapsed_time(e1) for e0, e1 in evs) / ITERS_WARM * 1e-3

                evs = [(torch.cuda.Event(enable_timing = True),
                        torch.cuda.Event(enable_timing = True)) for _ in range(ITERS_COLD)]
                for e0, e1 in evs:
                    thrash.add_(1.0)
                    e0.record()
                    step()
                    e1.record()
                torch.cuda.synchronize(device)
                lat_c = sum(e0.elapsed_time(e1) for e0, e1 in evs) / ITERS_COLD * 1e-3

            gb = (w_bytes + B * S * row_bytes) / 1e9
            mf = B * S * row_flops / 1e6
            print(f"{B:>2} {S:>2} | {lat_w * 1e6:>8.1f} {lat_c * 1e6:>8.1f} "
                  f"{B * S / lat_c:>9.0f} | "
                  f"{gb:>9.4f} {gb / lat_c:>8.1f} | {mf:>8.0f} {mf / 1e3 / lat_c:>10.1f}")

            for rs in states:
                rs.free()

if __name__ == "__main__":
    main()
