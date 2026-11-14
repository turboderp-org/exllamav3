import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import heapq
from collections import defaultdict

"""
Compile a per-tensor quantization recipe from a sc_measure.py sensitivity measurement.

Each tensor's end-to-end KL contribution at bitrate K is modeled as

    kld(t, K) = S_t * rfn_t(K)^alpha

where S_t is the measured sensitivity (kld / rfn^alpha at the injection level), alpha is the
scaling exponent fitted across the measurement's noise levels (~2, the quadratic law), and
rfn_t(K) is the expected weight-space quantization error, following the EXL3 trellis error
curve: amplitude halves per bit, anchored either to a per-tensor rfn probe of a real quantized
model (--rfn_ref) or to a global anchor (--anchor).

Since kld(t, K) is convex and decreasing in K while storage cost is linear (numel bits per K),
the budget allocation is a separable convex problem: greedy assignment by marginal KL reduction
per bit is exact (up to one tensor of granularity). No end-to-end measurements of candidate
quantized models are needed, and no combination of pre-quantized checkpoints is compiled: the
output is only a recipe, so a subsequent convert.py run can LDLQ-compensate against the actual
noise of the tensors as quantized.

Output: YAML recipe mapping each quantizable tensor to a bitrate, plus head_bits,
intended to be consumed by convert.py in place of --bits/--head_bits.
"""


# Half-integer trellis rates with kernel instances (mul1 codebook only); above 3.5 bpw the steps are whole bits
HALF_RATES = (1.5, 2.5, 3.5)


def main(args):
    with open(args.measurement, "r") as f:
        meas = json.load(f)

    results = meas["results"]
    tensors = [r for r in results if r["qbits_key"] == "bits"]
    head = [r for r in results if r["qbits_key"] == "head_bits"]

    # Scaling exponent kld ~ rfn^alpha, fitted per tensor across noise levels, median over model
    if args.alpha:
        alpha = args.alpha
    else:
        alphas = []
        for r in results:
            l0, l1 = r["levels"][0], r["levels"][-1]
            if l0["kld"] > 0 and l1["kld"] > 0 and l0["rfn_actual"] != l1["rfn_actual"]:
                alphas.append(
                    math.log(l0["kld"] / l1["kld"]) / math.log(l0["rfn_actual"] / l1["rfn_actual"]))
        assert alphas, "measurement has a single noise level; pass --alpha explicitly"
        alphas.sort()
        alpha = alphas[len(alphas) // 2]
    print(f" -- Scaling exponent: {alpha:.3f}")

    # Expected quantization error per tensor per K: anchored curve, amplitude ratio per bit
    ratio = args.bit_ratio
    if args.rfn_ref:
        with open(args.rfn_ref, "r") as f:
            ref = json.load(f)
        anchor = {r["key"]: (r["K"], r["rfn"]) for r in ref["results"]}
    else:
        k0, r0 = args.anchor.split(":")
        anchor = defaultdict(lambda: (int(k0), float(r0)))

    def rfn_at(key, k):
        ak, ar = anchor[key]
        return ar * ratio ** (ak - k)

    # Sensitivity: geometric mean of kld / rfn^alpha over measured levels
    def sensitivity(r):
        s, n = 0.0, 0
        for l in r["levels"]:
            if l["kld"] > 0:
                s += math.log(l["kld"] / l["rfn_actual"] ** alpha)
                n += 1
        return math.exp(s / n) if n else 0.0

    # Grouped targets (a MoE layer's routed experts, "members" in the measurement) anchor on
    # the rms of their members' reference errors at the members' common K
    if args.rfn_ref:
        for r in tensors + head:
            if "members" in r and r["key"] not in anchor:
                missing = [m for m in r["members"] if m not in anchor]
                if missing:
                    raise ValueError(f"{missing[0]} (member of {r['key']}) missing from {args.rfn_ref}")
                ks = [anchor[m][0] for m in r["members"]]
                k_common = max(set(ks), key = ks.count)
                rms = math.sqrt(sum(anchor[m][1] ** 2 for m in r["members"]) / len(r["members"]))
                anchor[r["key"]] = (k_common, rms)

    for r in tensors + head:
        r["S"] = sensitivity(r)
        if r["key"] not in anchor and args.rfn_ref:
            raise ValueError(f"{r['key']} missing from {args.rfn_ref}")

    # Below the anchor K (demotion territory, rfn > ~0.3) the observed KL-vs-noise law is
    # steeper than the in-range fit: on qwen3.8-27b the K2->K1 octave ran at an effective
    # exponent ~2.35 vs ~2.0 for K3->K2 (measured against real-quant swap attribution, partly
    # inflated by dataset mismatch in that pipeline). --alpha_low applies only to the portion of
    # the curve below the anchor, leaving in-range predictions untouched
    alpha_low = args.alpha_low if args.alpha_low is not None else alpha

    def kld_at(r, k):
        key = r["key"]
        ak, _ = anchor[key]
        base = r["S"] * rfn_at(key, max(k, ak)) ** alpha
        if k < ak:
            base *= (rfn_at(key, k) / rfn_at(key, ak)) ** alpha_low
        return base

    # Tie groups: tensors fused into a single GEMM by the fast inference paths (the sliced
    # q/k/v bundle, gate/up, the GDN qkv/z bundle) must share a bitrate. Tied by key suffix
    # within the same parent module; --tie "" or unmatched suffixes leave tensors independent.
    # Measurements made with grouped targets already carry these as single results
    tie_sets = [set(part.split("+")) for part in args.tie.split(",")] if args.tie.strip() else []

    def group_of(key):
        parent, _, leaf = key.rpartition(".")
        for gi, s in enumerate(tie_sets):
            if leaf in s:
                return (parent, gi)
        return key

    group_map = {}
    for r in tensors:
        group_map.setdefault(group_of(r["key"]), []).append(r)
    groups = list(group_map.values())
    num_tied = sum(1 for g in groups if len(g) > 1)
    if num_tied:
        print(f" -- Tied groups: {num_tied} ({args.tie})")

    # Supported bitrates within [min_k, max_k], ascending: whole bits, plus the half-integer rates with --k_step 0.5
    all_rates = sorted(set(range(1, 9)) | (set(HALF_RATES) if args.k_step == 0.5 else set()))
    assert args.min_k in all_rates and args.max_k in all_rates, \
        f"--min_k and --max_k must be supported bitrates: {', '.join(str(k) for k in all_rates)}"
    rates = [k for k in all_rates if args.min_k <= k <= args.max_k]
    assert rates, "--min_k is above --max_k"

    # Exact greedy: all groups start at min_k, spend budget one supported increment at a time on the largest
    # marginal KL reduction per storage bit. Increments differ in size (half a bit up to 4 bpw, a whole bit above),
    # which the per-bit ranking accounts for; the curve is convex, so each group's increments still rank in order
    sum_numel = sum(r["numel"] for r in tensors)
    budget = int(args.bitrate * sum_numel)
    numel_g = [sum(r["numel"] for r in g) for g in groups]
    level = [0] * len(groups)                               # index into rates, per group
    spent = rates[0] * sum_numel
    assert spent <= budget, f"target bitrate below min_k = {rates[0]}"

    def kld_at_g(g, k):
        return sum(kld_at(r, k) for r in g)

    # KL reduction and storage cost of raising group i from rates[l] to rates[l + 1]
    def step_gain(i, l):
        return kld_at_g(groups[i], rates[l]) - kld_at_g(groups[i], rates[l + 1])

    def step_cost(i, l):
        return numel_g[i] * (rates[l + 1] - rates[l])

    heap = []
    def push(i, l):
        if l + 1 < len(rates):
            heapq.heappush(heap, (-step_gain(i, l) / step_cost(i, l), i, l))

    for i in range(len(groups)):
        push(i, 0)
    while heap:
        _, i, l = heapq.heappop(heap)
        if spent + step_cost(i, l) > budget:
            # Later increments for this group only get worse per bit; drop it and let smaller
            # groups keep filling the remainder
            continue
        level[i] = l + 1
        spent += step_cost(i, l)
        push(i, l + 1)

    # Exchange repair: a group (especially a tied pair, whose increments are twice the size) can
    # be stranded at a low K when its next increment stops fitting the remaining budget while
    # smaller groups keep filling. Trade: fund one promotion by demoting the cheapest-loss-per-bit
    # groups (possibly several — post-greedy leftover is always below the smallest increment, so
    # a pair-sized promotion typically needs two demotions) whenever the exchange lowers
    # predicted KLD within the budget. Each accepted move strictly lowers KLD so this
    # terminates; the cap is a safety net
    exchanges = 0
    for _ in range(100):
        leftover = budget - spent
        # Demotion candidates, cheapest loss per freed bit first, shared across promotions
        cands = []
        for j in range(len(groups)):
            if level[j] > 0:
                loss, freed = step_gain(j, level[j] - 1), step_cost(j, level[j] - 1)
                cands.append((loss / freed, loss, freed, j))
        cands.sort()
        best = None  # (kld improvement, promote group, [demote groups])
        for i in range(len(groups)):
            if level[i] + 1 >= len(rates):
                continue
            gain = step_gain(i, level[i])
            need = step_cost(i, level[i]) - leftover
            demote, loss_sum = [], 0.0
            for _, loss, freed, j in cands:
                if need <= 0:
                    break
                if j == i:
                    continue
                demote.append(j)
                loss_sum += loss
                need -= freed
            if need > 0:
                continue
            delta = gain - loss_sum
            if delta > 0 and (best is None or delta > best[0]):
                best = (delta, i, demote)
        if best is None:
            break
        _, i, demote = best
        spent += step_cost(i, level[i])
        level[i] += 1
        for j in demote:
            level[j] -= 1
            spent -= step_cost(j, level[j])
        exchanges += 1
    if exchanges:
        print(f" -- Exchange repair: {exchanges} move(s)")

    k_assign = {r["key"]: rates[level[i]] for i, g in enumerate(groups) for r in g}
    achieved_bpw = spent / sum_numel
    pred_kld = sum(kld_at(r, k_assign[r["key"]]) for r in tensors)

    # Approximation of the default convert.py strategy for comparison: floor bitrate everywhere,
    # promote tensors to ceil starting from the ends of the stack
    base_k = max(args.min_k, min(args.max_k, int(math.floor(args.bitrate))))
    k_base = {r["key"]: base_k for r in tensors}
    layers = [l for l in (layer_of(r["key"]) for r in tensors) if l is not None]
    max_layer = max(layers) if layers else 0
    def end_dist(r):
        l = layer_of(r["key"])
        return 0 if l is None else min(l, max_layer - l)
    spent_b = sum(base_k * r["numel"] for r in tensors)
    for r in sorted(tensors, key = end_dist):
        if base_k >= args.max_k:
            break
        if spent_b + r["numel"] <= budget:
            k_base[r["key"]] = base_k + 1
            spent_b += r["numel"]
    pred_kld_base = sum(kld_at(r, k_base[r["key"]]) for r in tensors)

    print(f" -- Budget: {budget:,} bits over {sum_numel:,} weights")
    print(f" -- Achieved bitrate:      {achieved_bpw:.4f} bpw (target {args.bitrate:.4f})")
    print(f" -- Predicted KLD:         {pred_kld:.6f}")
    print(f" -- Predicted KLD (ends-first baseline at same budget): {pred_kld_base:.6f}")
    if head:
        print(f" -- Predicted head KLD at K={args.head_bits}: {kld_at(head[0], args.head_bits):.6f}")

    print("\n -- Bitrate histogram:")
    hist = defaultdict(lambda: [0, 0])
    for r in tensors:
        h = hist[k_assign[r["key"]]]
        h[0] += 1
        h[1] += r["numel"]
    for k in sorted(hist):
        n, numel = hist[k]
        print(f"      K={k}: {n:4} tensors  {numel:15,} weights  ({100 * numel / sum_numel:5.1f}%)")

    print("\n -- By module type:")
    by_type = defaultdict(list)
    for r in tensors:
        by_type[r["key"].split(".")[-1]].append(k_assign[r["key"]])
    for t, ks in sorted(by_type.items()):
        print(f"      {t:20} min {min(ks)}  max {max(ks)}  mean {sum(ks) / len(ks):.2f}")

    # Recipe
    recipe = dict(
        model = meas["model"],
        measurement = args.measurement,
        target_bpw = args.bitrate,
        achieved_bpw = round(achieved_bpw, 4),
        predicted_kld = round(pred_kld, 6),
        head_bits = args.head_bits,
        tie = args.tie,
        # A grouped target's bitrate applies to every member tensor (one K per MoE layer's experts)
        tensors = {m: k_assign[r["key"]] for r in tensors for m in r.get("members", [r["key"]])},
    )
    with open(args.out, "w") as f:
        f.write("# Quantization recipe generated by sc_optimize.py\n")
        for k, v in recipe.items():
            if k == "tensors":
                f.write("tensors:\n")
                for key, bits in v.items():
                    f.write(f"  {key}: {bits}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"\n -- Saved: {args.out}")

    # Optional full predicted-KL table, (num_linears, K) for external solvers
    if args.table:
        table = dict(
            model = meas["model"],
            alpha = alpha,
            bit_ratio = ratio,
            k_range = [rates[0], rates[-1]],
            tensors = [
                dict(
                    key = r["key"],
                    qbits_key = r["qbits_key"],
                    numel = r["numel"],
                    S = r["S"],
                    kld = {k: kld_at(r, k) for k in rates},
                )
                for r in tensors + head
            ],
        )
        with open(args.table, "w") as f:
            json.dump(table, f, indent = 2)
        print(f" -- Saved: {args.table}")


def layer_of(key):
    for part in key.split("."):
        if part.isdigit():
            return int(part)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-m", "--measurement", type = str, required = True, help = "Sensitivity measurement JSON from sc_measure.py")
    parser.add_argument("-b", "--bitrate", type = float, required = True, help = "Target mean bitrate over quantizable (non-head) weights")
    parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Head bitrate, default: 6")
    parser.add_argument("-rr", "--rfn_ref", type = str, default = None, help = "Per-tensor quantization error anchors (rfn probe JSON)")
    parser.add_argument("-a", "--anchor", type = str, default = "2:0.292", help = "Global error anchor K:rfn if no --rfn_ref, default: 2:0.292")
    parser.add_argument("-br", "--bit_ratio", type = float, default = 1.96, help = "Quantization error amplitude ratio per bit, default: 1.96")
    parser.add_argument("-al", "--alpha", type = float, default = None, help = "Override scaling exponent (default: fit from measurement)")
    parser.add_argument("-all", "--alpha_low", type = float, default = None, help = "Scaling exponent for the curve below the anchor K (demotions to high noise, e.g. K=1); default: same as --alpha.")
    parser.add_argument("-mink", "--min_k", type = float, default = 1)
    parser.add_argument("-maxk", "--max_k", type = float, default = 8)
    parser.add_argument("-ks", "--k_step", type = float, default = 1.0, choices = (1.0, 0.5),
                        help = "Bitrate increment of the allocation: 1 (default) or 0.5 (adds the half-integer trellis rates 1.5, 2.5 and 3.5; whole bits above "
                               "those). Half-integer rates need the mul1 codebook")
    parser.add_argument("-tie", "--tie", type = str, default = "q_proj+k_proj+v_proj,gate_proj+up_proj,in_proj_qkv+in_proj_z",
                        help = "Suffix groups forced to share a bitrate (fused GEMMs in the inference paths), tied within the same parent module. Pass \"\" to optimize all "
                               "tensors independently. Default: k_proj+v_proj,gate_proj+up_proj")
    parser.add_argument("-o", "--out", type = str, required = True, help = "Output recipe (YAML)")
    parser.add_argument("-t", "--table", type = str, default = None, help = "Optionally save full predicted per-tensor per-K KLD table (JSON)")
    main(parser.parse_args())
