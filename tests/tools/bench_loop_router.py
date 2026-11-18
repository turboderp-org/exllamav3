"""GPU router microbenchmark: python tests/tools/bench_loop_router.py."""
import os
import pathlib
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from exllamav3.modules.loop_router import route


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.manual_seed(42)
    streams, hidden = 2, 4096
    weight = torch.randn(8, streams * hidden, device = "cuda", dtype = torch.float32) * 0.01
    bias = torch.randn(8, device = "cuda", dtype = torch.float32)
    for rows in (1, 4):
        x = torch.randn(rows, streams, hidden, device = "cuda", dtype = torch.float16)
        for label, enabled in (("torch", "0"), ("triton", "1")):
            os.environ["EXL3_LOOP_ROUTER_TRITON"] = enabled
            def call():
                return route(x, weight, bias, streams, 20, False, 1.0)
            call()
            ms = triton.testing.do_bench(call)
            print(f"rows={rows} {label}: {ms * 1000:.1f} us/call")


if __name__ == "__main__":
    main()
