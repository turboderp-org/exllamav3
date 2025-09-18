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
    q: Linear | None,
    k: Linear | None,
    v: Linear | None,
    o: Linear | None,
    g: Linear | list[Linear] | None,
    u: Linear | list[Linear] | None,
    d: Linear | list[Linear] | None,
    qkvz: Linear | None,
) -> (dict, int):

    # Submodules
    keys = []
    out_keys = {}
    numels = []
    perms_qkvo = []
    perms_gud = []

    if q is not None:
        assert k and v and o
        keys += [m.key for m in (q, k, v, o)]
        numels += [m.weights_numel() for m in (q, k, v, o)]
        for m in (q, k, v, o):
            out_keys[m.key] = m.key
        perms_qkvo = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 1, 1, 1],
            [0, 1, 2, 1],
            [1, 2, 2, 1],
        ]

    if qkvz is not None:
        assert o
        keys += [m.key for m in (qkvz, o)]
        numels += [m.weights_numel() for m in (qkvz, o)]
        for m in (qkvz, o):
            out_keys[m.key] = m.key
        perms_qkvo = [
            [0, 0],
            [1, 1],
            [2, 2],
        ]

    if g is not None and u is not None:
        assert d
        if isinstance(g, list):
            for m in (g, u, d):
                key_ = m[0].key.replace(".slice.0", ".slice.*").replace(".experts.0.", ".experts.*.")
                keys += [key_]
                numels += [sum(mm.weights_numel() for mm in m)]
                for mm in m:
                    out_keys[mm.key] = key_
        else:
            keys += [m.key for m in (g, u, d)]
            numels += [m.weights_numel() for m in (g, u, d)]
            for m in (g, u, d):
                out_keys[m.key] = m.key
        perms_gud = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]

    elif g is None and u is not None:
        assert d
        if isinstance(u, list):
            for m in (u, d):
                key_ = m[0].key.replace(".slice.0", ".slice.*").replace(".experts.0.", ".experts.*.")
                keys += [key_]
                numels += [sum(mm.weights_numel() for mm in m)]
                for mm in m:
                    out_keys[mm.key] = key_
        else:
            keys += [m.key for m in (u, d)]
            numels += [m.weights_numel() for m in (u, d)]
            for m in (u, d):
                out_keys[m.key] = m.key
        perms_gud = [
            [0, 0],
            [0, 1],
            [1, 1],
        ]

    # Bits per weight from budget
    numel = sum(numels)
    budget = int(bpw * numel) + surplus_bits + 1
    bpw = budget / numel

    # Permutations to consider
    base_bpw = max(int(math.floor(bpw)), 1)
    if perms_qkvo and perms_gud:
        perms = [qkvo + gud for qkvo in perms_qkvo for gud in perms_gud]
        perms = [[min(8, p1 + base_bpw) for p1 in p2] for p2 in perms]
    elif perms_qkvo:
        perms = perms_qkvo
        perms = [[min(8, p1 + base_bpw) for p1 in p2] for p2 in perms]
    elif perms_gud:
        perms = perms_gud
        perms = [[min(8, p1 + base_bpw) for p1 in p2] for p2 in perms]
    else:
        assert False, "Logic error"

    # Find largest option within budget
    options = [(sum(a * b for a, b in zip(p, numels)), p) for p in perms]
    options.sort()
    idx = bisect.bisect_right(options, (budget,))
    idx = max(0, idx - 1)
    used_budget, selected = options[idx]

    # Output
    strategy = {k: v for k, v in zip(keys, selected)}
    strategy = {k: strategy[v] for k, v in out_keys.items()}
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
