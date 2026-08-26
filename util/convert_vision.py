import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import json
import time
from exllamav3 import Model, Config
from exllamav3.util import Timer
from exllamav3.modules.quant import LinearEXL3, LinearFP16
from exllamav3.modules import Linear
from exllamav3.loader.safetensors_alt import save_file
from exllamav3.conversion.compile import dsize
from pathlib import Path

"""
Utility to (re)convert just the vision component of an already-converted model, exporting the
quantized tower as a donor-style model directory (model.safetensors + config.json with
quantization_config->vision_bits). Combine with util/transplant_vision.py to graft the result
onto the source checkpoint:

    python util/convert_vision.py -m <model_dir> -vb 6 -o <donor_dir>
    python util/transplant_vision.py -m <model_dir> -d <donor_dir> -o <output_dir>

Vision towers are quantized without calibration (identity Hessian), like MTP heads: inference
is a single compute-bound batch per image and the towers tolerate 5-6 bpw with no measurable
loss. -vb 16 copies the tower unquantized (e.g. to strip a quantized tower back to fp16).
"""


def quantize_linears_single(bitrate, device, linears, config):
    for linear in linears:
        if bitrate == 16:
            print(
                f" -- Unquantized: {linear.key:{config.stc.max_key_len() + 6}}"
                f"  bpw: {16:5.2f}",
                flush = True
            )
        else:
            quant_args = {
                "seed": 0,
                "mul1": True,
                "K": bitrate,
                "devices": [device],
                "device_ratios": None,
                "apply_out_scales": "always",
            }
            with Timer() as t:
                proxy_err = linear.convert_exl3(
                    linear.init_H_data(False),
                    quant_args = quant_args,
                    progress_str = f" -- <step>: {linear.key}",
                    verbose = False,
                    save_reg = None,
                )
                assert isinstance(linear.inner, LinearEXL3)
                linear.inner.swap_cpu()
            flags = "o" if quant_args["apply_out_scales"] else "."
            flags += "f" if quant_args["q_fallback"] else "."
            proxy_err_str = (
                "(zero)  " if quant_args["zeros"] else
                "(big)   " if proxy_err >= 9.9 else
                f"{proxy_err:8.6f}" if proxy_err >= 0.0 else
                "(OoM)   "
            )
            proxy_err_label = "proxy_err" if not quant_args["q_fallback"] else "rmse"
            print(
                f" -- Quantized: {linear.key:{config.stc.max_key_len() + 8}}"
                f"  bpw: {quant_args['K']:5.2f}"
                f"  {proxy_err_label}: {proxy_err_str}"
                f"  {flags}"
                f"  g_sc: {quant_args['g_scale']:.6f}"
                f"  [{t.interval:4.2f} s]",
                flush = True
            )


def feedback_module(module, config, final_bpw, module_time):
    print(
        f" -- Finished: {module.key:{config.stc.max_key_len() + 8}}" +
        (f"  bpw: {final_bpw:5.2f}" if final_bpw else f"  no_weights") +
        f"  [{module_time:.2f} s]",
        flush = True
    )


def main(args):

    config = Config.from_directory(args.model_dir)
    assert "vision" in config.model_classes, \
        f"{config.architecture} has no vision component"
    vmodel = Model.from_config(config, component = "vision")
    q_tensors = {}
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)
    assert not any(f.suffix == ".safetensors" for f in out_dir.iterdir()), \
        f"Output directory already contains a model: {out_dir}"

    # -vb 16: raw copy of the source tower (no module round-trip: padded architectures
    # would otherwise re-emit padded fp16 weights)
    modules = vmodel.modules if args.vision_bits != 16 else []

    for idx, module in enumerate(modules):
        assert module.num_slices <= 1
        start_module_time = time.time()

        module_bytes_before = dsize(q_tensors)
        print(f" -- Loading unquantized module: {module.key}")
        defer = module.can_defer_load()
        if defer:
            config.stc.begin_deferred_load()
        try:
            module.load(torch.device("cpu" if module.caps.get("prefer_cpu") else args.device))
        finally:
            if defer:
                config.stc.end_deferred_load()
        for m in module:
            if m.used_alt_key:
                print(f"     - Cloned {m.key} from {m.alt_key}")
        module.config.stc.close()

        linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]
        assert all(isinstance(linear.inner, LinearFP16) for linear in linears), \
            f"Vision component of {args.model_dir} is already quantized; " \
            f"quantize from a checkpoint with an fp16 tower"

        # Move original tensors to system RAM (load to GPU one by one when quantizing)
        for linear in linears:
            linear.inner.swap_cpu()

        # Quantize
        quantize_linears_single(args.vision_bits, args.device, linears, config)

        # Collect converted module tensors
        for m in module:
            q_tensors.update(m.get_tensors())

        # Unload module
        module.unload()
        config.stc.close()

        # Output final bpw for layer (bytes added by THIS module; q_tensors is cumulative)
        num_bytes = dsize(q_tensors) - module_bytes_before
        num_bits = num_bytes * 8
        final_bpw = num_bits / module.weights_numel() if module.weights_numel() else None

        feedback_module(module, config, final_bpw, time.time() - start_module_time)
        module.unload()

    # Aux vision tensors not owned by the module tree (position embeddings, runtime-fetched
    # extras): everything the compile step would enumerate for the vision component, minus
    # what the quantized modules produced. The .weight of a quantized linear is replaced by
    # its EXL3 tensors, so skip any name whose parent key now owns a trellis
    all_vision = config.model_classes["vision"].get_additional_compiled_tensors(config)
    quantized_parents = {k.rsplit(".", 1)[0] for k in q_tensors if k.endswith(".trellis")}
    n_aux = 0
    for name in all_vision:
        if name in q_tensors:
            continue
        if name.rsplit(".", 1)[0] in quantized_parents:
            continue
        q_tensors[name] = config.stc.get_tensor(name, "cpu")
        n_aux += 1
    config.stc.close()
    print(f" -- Copied {n_aux} auxiliary vision tensors unmodified")

    # Save collected output as a donor-style directory: single shard named for
    # transplant_vision.py's shard_map, plus config.json carrying vision_bits
    out_st = out_dir / "model.safetensors"
    print(f" -- Writing {out_st}")
    save_file(q_tensors, out_st)

    config_dict = json.load(open(os.path.join(args.model_dir, "config.json")))
    qcfg = config_dict.setdefault("quantization_config", {})
    if args.vision_bits != 16:
        qcfg["vision_bits"] = args.vision_bits
    else:
        qcfg.pop("vision_bits", None)
    with open(out_dir / "config.json", "w") as f:
        f.write(json.dumps(config_dict, indent = 4))

    # Preprocessing configs, so the donor is loadable standalone as a vision component (e.g.
    # for validating the quantized tower before transplanting)
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json",
               "video_preprocessor_config.json", "chat_template.json"):
        src = os.path.join(args.model_dir, fn)
        if os.path.isfile(src):
            shutil.copy(src, out_dir / fn)

    print(" -- All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-m", "--model_dir", type = str, help = "Input model directory", required = True)
    parser.add_argument("-vb", "--vision_bits", type = int, help = "Vision tower bitrate, 1-8 or 16 for unquantized, default: 6", default = 6)
    parser.add_argument("-o", "--out_dir", type = str, help = "Output (donor) directory for the converted vision tower", required = True)
    parser.add_argument("-d", "--device", type = int, help = "Device index to use for quantization, default: 0", default = 0)
    _args = parser.parse_args()
    assert _args.vision_bits == 16 or 1 <= _args.vision_bits <= 8, \
        "--vision_bits must be 1-8, or 16 to store the vision model unquantized"
    main(_args)
