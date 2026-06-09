import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from exllamav3 import Model, Config
from exllamav3.util import Timer
from exllamav3.modules.quant import LinearEXL3, LinearFP16
from exllamav3.modules import Linear
from exllamav3.loader.safetensors_alt import save_file
from exllamav3.conversion.compile import dsize
from pathlib import Path
import time

"""
Utility to convert just the MTP model from a HF checkpoint and export the quantized tensors separately. A model already
quantized prior to MTP support being added will not contain the MTP tensors but can be augmented by placing the output
file from this script alongside the model's other .safetensors files.  
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
                "mcg": True,
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
        f" -- Quantized: {module.key:{config.stc.max_key_len() + 8}}" +
        (f"  bpw: {final_bpw:5.2f}" if final_bpw else f"  no_weights") +
        f"  [{module_time:.2f} s]",
        flush = True
    )


def main(args):

    # Load MTP component model
    config = Config.from_directory(args.model_dir)
    mtp_model = Model.from_config(config, component = "mtp")
    mtp_model.load(progressbar = True)
    q_tensors = {}
    out_path = Path(args.out_file).resolve()

    for idx, module in enumerate(mtp_model.modules):
        assert module.num_slices <= 1
        start_module_time = time.time()

        print(f" -- Loading unquantized module: {module.key}")
        module.load(torch.device("cpu" if module.caps.get("prefer_cpu") else args.device))
        for m in module:
            if m.used_alt_key:
                print(f"     - Cloned {m.key} from {m.alt_key}")
        module.config.stc.close()

        linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]
        assert all(isinstance(linear.inner, LinearFP16) for linear in linears)

        # Move original tensors to system RAM (load to GPU one by one when quantizing)
        for linear in linears:
            linear.inner.swap_cpu()

        # Quantize
        quantize_linears_single(args.mtp_bits, args.device, linears, config)

        # Collect converted module tensors
        for m in module:
            q_tensors.update(m.get_tensors())

        # Unload module
        module.unload()
        config.stc.close()

        # Output final bpw for layer
        num_bytes = dsize(q_tensors)
        num_bits = num_bytes * 8
        final_bpw = num_bits / module.weights_numel() if module.weights_numel() else None

        # Feedback after module
        module_time = time.time() - start_module_time
        feedback_module(module, config, final_bpw, module_time)

        # Unload current module
        module.unload()

    # Save collected output
    print(f" -- Writing {out_path}")
    save_file(q_tensors, out_path)

    # All done
    print(" -- All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type = str, help = "Input model directory", required = True)
    parser.add_argument("-mb", "--mtp_bits", type = int, help = "MTP model bitrate, default: 4", default = 4)
    parser.add_argument("-o", "--out_file", type = str, help = "Output .safetensors file to contain quantized MTP tensors")
    parser.add_argument("-d", "--device", type = int, help = "Device index to use for quantization, default: 0", default = 0)
    _args = parser.parse_args()
    main(_args)
