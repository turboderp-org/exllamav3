"""
DSV4-Flash-Vision-Exp 3.04bpw on sm70 port: autosplit load across all visible
GPUs + short greedy generation + decode timing.

    python tests/dsv4_flash_sm70_load_test.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import os
if os.environ.get("STUB_MGEMM") == "1":
    from exllamav3.ext import exllamav3_ext as _ext
    def _stub(*a, **k):
        raise RuntimeError("exl3_mgemm called (stubbed)")
    _ext.exl3_mgemm = _stub
if os.environ.get("STUB_BC") == "1":
    from exllamav3.ext import exllamav3_ext as _ext2
    class _StubBC:
        def __init__(self, *a, **k): pass
        def run_alloc(self, x, out_features, fp32):
            raise RuntimeError("bc.run_alloc called (stubbed)")
    _ext2.BC_LinearEXL3 = _StubBC
if os.environ.get("NO_COMP_BC") == "1":
    import exllamav3.modules.dsv4 as _dsv4
    from exllamav3.ext import exllamav3_ext as _extc
    def _patched_ff(self, x, params, buf_kv, buf_gate, ovl, dest_a, dest_b, position,
                    pool_bt = None, pool_epp = 0, stage_rel = False):
        if self.fused_norm_w is None or self.ape is None:
            return _orig_ff(self, x, params, buf_kv, buf_gate, ovl, dest_a, dest_b, position,
                           pool_bt, pool_epp, stage_rel)
        kv = self.wkv.forward(x, params)[0]
        gate = self.wgate.forward(x, params)[0]
        _extc.dsv4_compress(
            kv, gate, buf_kv, buf_gate, ovl, self.ape, self.fused_norm_w,
            self.norm.rms_norm_eps, self.fused_inv_freq, dest_a, dest_b, position,
            None, self.compress_rate, None, pool_bt, pool_epp, stage_rel)
    _orig_ff = _dsv4.DSV4Compressor.forward_fused
    _dsv4.DSV4Compressor.forward_fused = _patched_ff
    print("compressor BC disabled")

from exllamav3 import Config, Model, Cache, Tokenizer, Generator

model_dir = "/home/nvidia/Dev/model/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw"

config = Config.from_directory(model_dir)
model = Model.from_config(config)
print(f"Loading {model_dir} ...")
t0 = time.time()
cache = Cache(model, max_num_tokens = 8192, max_batch_size = 1)
model.config.infer_params.no_reconstruct = True
model.load(progressbar = True)
print(f"Loaded in {time.time() - t0:.1f} s; devices: {model.active_devices}")

tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
prompt = "The capital of Japan is"
response = generator.generate(
    prompt = prompt,
    max_new_tokens = 64,
    temperature = 0.0,
    top_p = 1.0,
    completion_only = True,
    add_bos = True,
)
print("Output:", response)

# Decode timing: 128 tokens greedy, timed
t0 = time.time()
generator.generate(
    prompt = "Write a detailed explanation of how a transformer language model works.",
    max_new_tokens = 128,
    temperature = 0.0,
    completion_only = True,
    add_bos = True,
)
dt = time.time() - t0
print(f"Decode: {128 / dt:.2f} tok/s ({dt:.1f} s for 128 tokens)")