from __future__ import annotations
import math
import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING
import re
if TYPE_CHECKING:
    from ..model import Model, Config

@dataclass
class QTarget:
    numel: int
    target_bpw: int
    min_bpw: int
    priority: int

    def total_bits(self):
        return self.numel * self.target_bpw

    def delta_1(self):
        delta = (min(8, self.target_bpw + 1) - self.target_bpw) * self.numel
        return delta

    def increase_1(self):
        self.target_bpw = min(8, self.target_bpw + 1)

    def clamp_min(self):
        self.target_bpw = max(self.target_bpw, self.min_bpw)


def create_q_strategy(
    model: Model,
    config: Config,
    bpw: float,
    head_bpw: int,
    hq: bool,
) -> (dict, float):
    from ..modules.module import Module
    from ..modules.linear import Linear

    base_bpw = int(math.floor(bpw))
    sum_numel = 0
    sum_bits = 0
    targets = {}
    aux_targets = {}
    target_groups = {}

    # Collect and initialize targets
    def _add(module: Module, priority: int):
        priority = max(priority, module.q_priority)
        nonlocal sum_numel, sum_bits, targets, aux_targets
        if isinstance(module, Linear) and module.qmap is not None:
            if module.qbits_key == "bits":
                numel = module.weights_numel()
                sum_numel += numel
                sum_bits += numel * base_bpw
                qt = QTarget(
                    numel = numel,
                    target_bpw = base_bpw,
                    min_bpw = min(base_bpw + module.select_hq_bits, 8),
                    priority = priority
                )
                targets[module.key] = qt
                if module.qgroup not in target_groups:
                    target_groups[module.qgroup] = []
                target_groups[module.qgroup].append(qt)

            elif module.qbits_key == "head_bits":
                numel = module.weights_numel()
                aux_targets[module.key] = QTarget(
                    numel = numel,
                    target_bpw = head_bpw,
                    min_bpw = head_bpw,
                    priority = priority
                )
            else:
                raise ValueError("Logic error in create_q_strategy")
        for sm in module.modules:
            _add(sm, priority)

    for m in model.modules:
        _add(m, 0)

    # Target
    max_bits = int(bpw * float(sum_numel))
    order = sorted(target_groups.values(), key = lambda x: max(y.priority for y in x), reverse = True)

    # Bump with priorities target met
    while sum_bits < max_bits:
        updates = False
        for target_list in order:
            cost = sum(t.delta_1() for t in target_list)
            if cost > 0 and sum_bits + cost <= max_bits:
                for t in target_list:
                    t.increase_1()
                updates = True
                sum_bits += cost
        if not updates:
            break

    # Apply constraint from --hq:
    final_bits = 0
    for t in targets.values():
        if hq:
            t.clamp_min()
        final_bits += t.total_bits()

    # Combined with head layer
    targets.update(aux_targets)

    # Keep only bitrate
    f_targets = {k: v.target_bpw for k, v in targets.items()}

    return f_targets, float(final_bits) / sum_numel


def print_strategy(
    strategy: dict
) -> str:
    pattern = re.compile(r'(?<=\.)\d+(?=\.)')
    output = {}
    max_length = 0
    for target_key, target_bpw in strategy.items():
        if target_bpw not in output:
            output[target_bpw] = {}
        mkey = pattern.sub("*", target_key)
        max_length = max(len(mkey), max_length)
        if mkey not in output[target_bpw]:
            output[target_bpw][mkey] = 0
        output[target_bpw][mkey] += 1

    r = ""
    for bpw, tensors in sorted(output.items()):
        r += f"     - {bpw} bpw:\n"
        for key, count in sorted(tensors.items()):
            r += f"        {key:{max_length}}   {count:8,} " + ("tensors\n" if count > 1 else "tensor\n")
    return r.rstrip()
