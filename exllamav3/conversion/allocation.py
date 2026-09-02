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
    mtp_model: Model,
    config: Config,
    bpw: float,
    head_bpw: int,
    mtp_bpw: int,
    hq: bool,
    vision_model: Model = None,
    vision_bpw: int = None,
) -> (dict, float):
    """
    Build the per-module quantization bitrate strategy for a converted model.

    Quantizable modules declare their quantization role in the model architecture, primarily through their qmap,
    qbits_key, qgroup, q_priority and select_hq_bits attributes. This function walks the module tree, aggregates
    all eligible Linear layers into qgroups, assigns an initial integer bitrate from the requested average bpw and
    then spends the remaining bit budget one bit at a time according to group priority. Auxiliary targets, such as
    output heads using head_bits, are collected alongside the main budgeted weights and merged into the returned
    strategy.
    """
    from ..modules.module import Module
    from ..modules.linear import Linear

    base_bpw = int(math.floor(bpw))
    sum_numel = 0
    sum_bits = 0
    targets = {}
    aux_targets = {}
    target_groups = {}

    # Collect and initialize targets
    def _add(module: Module, priority: int, fixed_bpw: int = None):
        priority = max(priority, module.q_priority)
        nonlocal sum_numel, sum_bits, targets, aux_targets
        if isinstance(module, Linear) and module.qmap is not None:
            if fixed_bpw is not None:
                # Uncalibrated side model (vision tower): every eligible Linear gets the same
                # fixed bitrate outside the main budget
                numel = module.weights_numel()
                aux_targets[module.key] = QTarget(
                    numel = numel,
                    target_bpw = fixed_bpw,
                    min_bpw = fixed_bpw,
                    priority = priority
                )

            elif module.qbits_key == "bits":
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

            elif module.qbits_key == "mtp_bits":
                numel = module.weights_numel()
                aux_targets[module.key] = QTarget(
                    numel = numel,
                    target_bpw = mtp_bpw,
                    # --hq promotes the same select layers (attention, shared experts) inside
                    # the MTP head as in the trunk; 16 = store unquantized, never promoted
                    min_bpw = min(mtp_bpw + module.select_hq_bits, 8) if mtp_bpw <= 8 else mtp_bpw,
                    priority = priority
                )

            else:
                raise ValueError("Logic error in create_q_strategy")
        for sm in module.modules:
            _add(sm, priority, fixed_bpw)

    modules = model.modules
    if mtp_model:
        modules = modules + mtp_model.modules
    for m in modules:
        _add(m, 0)
    if vision_model:
        for m in vision_model.modules:
            _add(m, 0, fixed_bpw = vision_bpw)

    # Target
    max_bits = int(bpw * float(sum_numel))

    # Promotion order: group priority first, then distance to the nearer end of the forward pass.
    # End layers contribute disproportionately to end-to-end error
    stack_max = {}
    group_meta = []
    for idx, (gkey, tlist) in enumerate(target_groups.items()):
        m = re.search(r"\.(\d+)\.", gkey)
        stack, layer = (gkey[:m.start()], int(m.group(1))) if m else (None, -1)
        if stack is not None:
            stack_max[stack] = max(stack_max.get(stack, -1), layer)
        group_meta.append((tlist, stack, layer, idx))

    def group_order_key(meta):
        tlist, stack, layer, idx = meta
        dist = 0 if stack is None else min(layer, stack_max[stack] - layer)
        return (-max(t.priority for t in tlist), dist, layer, idx)

    order = [meta[0] for meta in sorted(group_meta, key = group_order_key)]

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

    # Aux targets sit outside the main budget; the only ones with min_bpw above their target
    # are MTP-head layers with select_hq_bits (head/vision targets clamp as a no-op)
    if hq:
        for t in aux_targets.values():
            t.clamp_min()

    # Combined with head layer
    targets.update(aux_targets)

    # Keep only bitrate
    f_targets = {k: v.target_bpw for k, v in targets.items()}

    return f_targets, float(final_bits) / sum_numel if sum_numel else 0


def create_q_strategy_from_recipe(
    model: Model,
    mtp_model: Model,
    config: Config,
    recipe_tensors: dict,
    head_bpw: int,
    mtp_bpw: int,
    vision_model: Model = None,
    vision_bpw: int = None,
) -> (dict, float):
    """
    Build the per-module quantization strategy from an explicit per-tensor recipe (e.g. produced
    by util/sc_optimize.py) instead of the budgeted allocation in create_q_strategy. recipe_tensors
    maps each budgeted ("bits") Linear's key to an integer bitrate (1-8, or 16 to store unquantized).
    The recipe must cover the budgeted tensors exactly.
    """
    from ..modules.linear import Linear

    targets = {}
    sum_numel = 0
    sum_bits = 0
    missing = []

    def _add(module, fixed_bpw = None):
        nonlocal sum_numel, sum_bits
        if isinstance(module, Linear) and module.qmap is not None:
            if fixed_bpw is not None:
                targets[module.key] = fixed_bpw
            elif module.qbits_key == "bits":
                bpw = recipe_tensors.get(module.key)
                if bpw is None:
                    missing.append(module.key)
                else:
                    targets[module.key] = bpw
                    if bpw <= 8:
                        sum_numel += module.weights_numel()
                        sum_bits += module.weights_numel() * bpw
            elif module.qbits_key == "head_bits":
                targets[module.key] = head_bpw
            elif module.qbits_key == "mtp_bits":
                targets[module.key] = mtp_bpw
            else:
                raise ValueError("Logic error in create_q_strategy_from_recipe")
        for sm in module.modules:
            _add(sm, fixed_bpw)

    modules = model.modules
    if mtp_model:
        modules = modules + mtp_model.modules
    for m in modules:
        _add(m)
    if vision_model:
        for m in vision_model.modules:
            _add(m, fixed_bpw = vision_bpw)

    if missing:
        raise ValueError(
            f"Recipe is missing bitrates for {len(missing)} quantizable tensor(s), "
            f"e.g.: {', '.join(missing[:5])}"
        )
    unknown = [k for k in recipe_tensors if k not in targets]
    if unknown:
        raise ValueError(
            f"Recipe contains {len(unknown)} key(s) not matching any budgeted tensor in the "
            f"model, e.g.: {', '.join(unknown[:5])}"
        )

    return targets, float(sum_bits) / sum_numel if sum_numel else 0


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
