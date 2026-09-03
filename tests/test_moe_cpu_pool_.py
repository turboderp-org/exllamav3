"""
CPU expert-offload worker pool (cpu/moe_mul1.cpp Pool): a dispatch publishes (generation, participant count) and
each participating worker must run exactly once per dispatch, with run() returning only after all of them have
finished. Regression for the non-atomic (gen, run_nw) pair: a worker preempted between the two loads could run
and ack the next dispatch twice, letting run() return early. Oversubscribes the CPU (2x hardware threads) and
alternates a small participant cap with the full pool, the configuration in which the race is reachable.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext


def test_pool_dispatch_exactly_once():
    threads = max(4, 2 * (os.cpu_count() or 4))
    iters = int(os.environ.get("EXL3_POOL_STRESS_ITERS", "30000"))
    anomalies = ext.exl3_moe_cpu_pool_stress(threads, iters, 2, 200)
    assert anomalies == 0, f"{anomalies} dispatch anomalies (double/missing runs or early return) in {iters} dispatches"


if __name__ == "__main__":
    test_pool_dispatch_exactly_once(); print("PASS")
