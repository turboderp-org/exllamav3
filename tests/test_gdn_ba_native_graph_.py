"""Check native split-GDN Graph replay against a fresh eager owner.

Run with --model pointing to an EXL3 checkpoint with split GDN, K5120/N96.
No test-only runtime API or Graph-node replacement is used.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from exllamav3 import Cache, Config, Model
from exllamav3.ext import exllamav3_ext as ext
from bench_gdn_ba_native import generate, target_layers


def new_owner(layer, rows):
    owner = ext.BC_GatedDeltaNetSplit(
        layer.qkv_proj.inner.bc, layer.z_proj.inner.bc, layer.o_proj.inner.bc,
        layer.ba_weight_t, layer.ba_bias, layer.dt_bias, layer.a_log,
        layer.num_k_heads, layer.num_v_heads, layer.k_head_dim, layer.v_head_dim,
        layer.conv1d_weight_flat, layer.conv1d_bias, layer.norm.bc, layer.beta_scale,
    )
    bundle = layer.multi_qkvz
    if bundle is not None:
        owner.set_qkvz_bundle(bundle.ptrs_trellis, bundle.ptrs_suh, bundle.ptrs_svh,
            bundle.meta, bundle.K, bool(bundle.mcg), bool(bundle.mul1))
    previous = layer.bc
    try:
        layer.bc = owner
        layer._bc_configure_slot(rows, 1, False)
    finally:
        layer.bc = previous
    return owner


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", required = True)
    parser.add_argument("--output", type = Path, required = True)
    parser.add_argument("--profile", action = "store_true")
    args = parser.parse_args()
    model = Model.from_config(Config.from_directory(args.model))
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    model.load(device = "cuda:0", progressbar = False)
    layer = target_layers(model)[0]
    saved = {}
    original = layer.forward

    def capture(x, params, *a, **kw):
        if x.shape[:2] == (1, 1) and not saved:
            state = params["recurrent_states"][0]
            recurrent_layer = state.cache.get_recurrent_layer((layer.layer_idx, params.get("layer_instance", 0)))
            conv, recurrent = recurrent_layer.get_state_tensors()
            saved.update(x = x.clone(), conv = conv[state.slot:state.slot + 1].clone(),
                recurrent = recurrent[state.slot:state.slot + 1].clone())
        return original(x, params, *a, **kw)

    checks = []
    profiling = False
    try:
        layer.forward = capture
        generate(model, cache, [1000] * 3000, 8)
        layer.forward = original
        assert saved, "native split-GDN decode was not exercised"
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            profiling = True
        for rows in (1, 2):
            owner = new_owner(layer, rows)
            x0 = saved["x"].repeat(rows, 1, 1)
            conv0 = saved["conv"].repeat(rows, *([1] * (saved["conv"].ndim - 1)))
            state0 = saved["recurrent"].repeat(rows, *([1] * (saved["recurrent"].ndim - 1)))
            slots = torch.arange(rows, device = x0.device, dtype = torch.int32)
            conv, state = conv0.clone(), state0.clone()
            allocations = []
            for step, factor in enumerate((1.0, 0.9, 1.1, 0.8, 1.2)):
                x = x0 * factor
                allocations.append(x)  # Keep old addresses alive; reuse cannot hide stale pointers.
                y = torch.empty_like(x)
                conv.copy_(conv0)
                state.copy_(state0)
                with torch.cuda.nvtx.range(f"ba_native:rows={rows}:step={step}"):
                    owner.run_bszN(x, y, conv, state, slots, False)
                actual = [t.clone() for t in (y, conv, state)]
                # A new owner's first call is the ordinary native eager path.
                reference = new_owner(layer, rows)
                ref_y, ref_conv, ref_state = torch.empty_like(x), conv0.clone(), state0.clone()
                reference.run_bszN(x, ref_y, ref_conv, ref_state, slots, False)
                for a, b in zip(actual, (ref_y, ref_conv, ref_state)):
                    assert a.isfinite().all()
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (rows, step)
                checks.append(dict(rows = rows, step = step, phase = "eager" if step == 0 else "capture" if step == 1 else "replay",
                    output_and_states_bitwise_equal = True))
            assert len({x.data_ptr() for x in allocations}) == len(allocations)
        torch.cuda.synchronize()
        result = dict(checks = checks, capability = torch.cuda.get_device_capability(),
            fast_path_device = torch.cuda.get_device_capability() == (12, 0),
            note = "Launch specialization and geometry must also be checked in the trace.")
        args.output.write_text(json.dumps(result, indent = 2) + "\n")
        print(json.dumps(result), flush = True)
    finally:
        layer.forward = original
        if profiling:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        model.unload()


if __name__ == "__main__":
    main()
