from __future__ import annotations
import math
import bisect
from functools import lru_cache
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules.linear import Linear

def allocate_transformer(
    bpw: float,
    surplus_bits: int,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    g: Linear,
    u: Linear,
    d: Linear,
) -> (dict, int):

    # Submodules
    keys = [
        q.key,
        k.key,
        v.key,
        o.key,
        g.key,
        u.key,
        d.key,
    ]
    numels = [
        q.weights_numel(),
        k.weights_numel(),
        v.weights_numel(),
        o.weights_numel(),
        g.weights_numel(),
        u.weights_numel(),
        d.weights_numel(),
    ]
    numel = sum(numels)

    # Bits per weight from budget
    budget = int(bpw * numel) + surplus_bits + 1
    bpw = budget / numel

    # Permutations to consider
    @lru_cache
    def get_perms(base):
        perms_qkvo = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 0],
            [0, 1, 1, 1],
            [0, 1, 2, 1],
            [1, 2, 2, 1],
        ]
        perms_gud = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        p = [qkvo + gud for qkvo in perms_qkvo for gud in perms_gud]
        p = [[min(8, p1 + base_bpw) for p1 in p2] for p2 in p]
        return p

    base_bpw = max(int(math.floor(bpw)), 1)
    perms = get_perms(base_bpw)

    # Find largest option within budget
    options = [(sum(a * b for a, b in zip(p, numels)), p) for p in perms]
    options.sort()
    idx = bisect.bisect_right(options, (budget,))
    idx = max(0, idx - 1)
    used_budget, selected = options[idx]

    # Output
    strategy = {k: v for k, v in zip(keys, selected)}
    surplus = budget - used_budget
    return strategy, surplus


def allocate_linear(
    bpw: float,
    surplus_bits: int,
    l: Linear,
) -> (dict, int):

    numel = l.weights_numel()
    budget = int(bpw * numel) + surplus_bits + 1
    bpw = budget / numel
    bpw = max(int(math.floor(bpw)), 1)
    used_budget = bpw * numel

    strategy = {l.key: bpw}
    surplus = budget - used_budget
    return strategy, surplus
