"""
CPU expert-offload worker pool (cpu/moe_mul1.cpp Pool): a dispatch publishes (generation, participant count) and
each participating worker must run exactly once per dispatch, with run() returning only after all of them have
finished. Regression for the non-atomic (gen, run_nw) pair: a worker preempted between the two loads could run
and ack the next dispatch twice, letting run() return early. Oversubscribes the CPU (2x hardware threads) and
alternates a small participant cap with the full pool, the configuration in which the race is reachable. The
extension's stress entry point counts the anomalies (double/missing runs, early returns); the contract is zero.
"""

import os

import pytest

from testlib.isolated import run_isolated

pytestmark = [pytest.mark.nogpu, pytest.mark.slow]

ITERATIONS = 30000


def _stress(threads, iterations):
    from exllamav3.ext import exllamav3_ext as ext
    return ext.exl3_moe_cpu_pool_stress(threads, iterations, 2, 200)


def test_pool_dispatch_exactly_once():
    threads = max(4, 2 * (os.cpu_count() or 4))
    # In a child process: the persistent pool never shrinks, so the oversubscribed pool would stay
    # behind in the test process, its surplus workers spinning on every later dispatch and pinned
    # onto the participants' cores, which stalls every later CPU MoE call in the session
    anomalies = run_isolated(_stress, threads, ITERATIONS)
    assert anomalies == 0, f"{anomalies} dispatch anomalies (double/missing runs or early return) in {ITERATIONS} dispatches"
