"""
ROCm MoE decode switches (RDNA decode op): with each switch, the cached decode of a MoE model must match the
default path, single-row and batched (concurrent jobs, so the MoE layers see 1, 3 and 6 rows per forward, the
5-6 row case being an MTP verify pass with 4 drafts). Every configuration runs in its own process, since the
switches are read once per process.

Contracts:

- EXL3_GEMV_MR_FUSE_IN=1 + EXL3_MOE_DEC_FUSE_IN=1 (input rotation fused into the dot kernels, same code on the same
  values): logits bitwise equal to the default path, provided the default path itself reproduces bitwise across
  processes (checked first; skipped otherwise, as then no bitwise contract can be stated).
- EXL3_ROCM_MOE_SH_DEC=1, EXL3_ROCM_MOE_SH_FOLD=1 (with EXL3_MOE_DEC_FUSE_IN=1), EXL3_MOE_DEC_ROWS=0 and
  EXL3_ROCM_MOE_DECODE=0 (generic cooperative kernels): different kernels or summation order, so the logits agree
  within the model's own noise floor (testlib.e2e.assert_logits_agree).
- Tensor parallelism over two devices, where the RDNA decode op serves the expert-parallel ranks: the default
  (EXL3_ROCM_MOE_EP=1) agrees with EXL3_ROCM_MOE_EP=0 (generic kernels on those ranks), and so does
  EXL3_ROCM_MOE_SH_FOLD=1 (shared expert split across the ranks).

Runs over every registry model with BlockSparseMLP layers (the "moe" roles and MoE stubs such as qwen4-exp-stub,
whose shared expert with a sigmoid gate is the case SH_FUSED / SH_DEC / SH_FOLD target); others skip. Switches
whose bounds a model's layers fall outside of (e.g. SH_FOLD needs an 8-bit shared expert) leave those
layers on the default path; the comparison still holds.
"""

import os

import pytest
import torch

from testlib.e2e import assert_logits_agree, gpu_split, load_model, noise_floor, prefill_decode_logits
from testlib.isolated import device_env, run_isolated

pytestmark = [pytest.mark.rocm_only, pytest.mark.models(), pytest.mark.slow]

DECODE_STEPS = 12
PROMPT_TOKENS = 144     # with the offsets below this fits the ~190-token reference text
BATCHES = (1, 3, 6)
OFFSET = 8              # token offset between the concurrent sequences

SINGLE = {
    "default": {},
    "default_again": {},
    "fuse_in": {"EXL3_GEMV_MR_FUSE_IN": "1", "EXL3_MOE_DEC_FUSE_IN": "1"},
    "sh_dec": {"EXL3_ROCM_MOE_SH_DEC": "1"},
    "sh_fold": {"EXL3_MOE_DEC_FUSE_IN": "1", "EXL3_ROCM_MOE_SH_FOLD": "1"},
    "dec_rows_off": {"EXL3_MOE_DEC_ROWS": "0"},
    "decode_off": {"EXL3_ROCM_MOE_DECODE": "0"},
}
AGREE = ("sh_dec", "sh_fold", "dec_rows_off", "decode_off")
TP = {
    "default": {},
    "ep_off": {"EXL3_ROCM_MOE_EP": "0"},
    "sh_fold": {"EXL3_MOE_DEC_FUSE_IN": "1", "EXL3_ROCM_MOE_SH_FOLD": "1"},
}


def decode_logits(model_dir: str, tp_devices: int, with_floor: bool) -> dict:
    """Child process: {"b<n>": (n * (DECODE_STEPS + 1), vocab) logits of n concurrent sequences} (+ "floor")"""
    with torch.inference_mode():
        if tp_devices:
            devices = [torch.device("cuda", i) for i in range(tp_devices)]
            ctx = load_model(model_dir, None, "-tp", "-gs", gpu_split(devices))
        else:
            ctx = load_model(model_dir, torch.device("cuda", 0))
        with ctx as lm:
            from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
            if not any(isinstance(m, BlockSparseMLP) for m in lm.model):
                return {"skip": f"{lm.config.architecture}: no BlockSparseMLP layers"}
            if tp_devices and not lm.model.caps.get("supports_tp"):
                return {"skip": f"{lm.config.architecture}: no tensor-parallel support"}
            need = PROMPT_TOKENS + OFFSET * (max(BATCHES) - 1)
            ids = lm.encode(max_tokens = need)
            if ids.shape[-1] < need:
                return {"skip": "reference text too short for the batch layout"}
            seqs = [ids[:, i * OFFSET : i * OFFSET + PROMPT_TOKENS] for i in range(max(BATCHES))]
            out = {}
            for n in BATCHES:
                logits = prefill_decode_logits(lm, seqs[:n] if n > 1 else seqs[0], DECODE_STEPS)
                out[f"b{n}"] = torch.cat(logits) if isinstance(logits, list) else logits
            if with_floor:
                out["floor"] = noise_floor(lm, seqs[0], DECODE_STEPS)
            return out


_runs = {}


def run(model_dir, devices, kind, config):
    """Cached per (model, single / tp, configuration)"""
    key = (model_dir, kind, config)
    if key not in _runs:
        if kind == "tp":
            env = {"CUDA_VISIBLE_DEVICES": tp_visible(devices[:2]), "EXL3_TEST_DEVICE": "cuda:0", **TP[config]}
            _runs[key] = run_isolated(decode_logits, model_dir, 2, config == "default", env = env)
        else:
            env = {**device_env(devices[0]), **SINGLE[config]}
            _runs[key] = run_isolated(decode_logits, model_dir, 0, config == "default", env = env)
        if "skip" in _runs[key]:
            pytest.skip(_runs[key]["skip"])
    return _runs[key]


def tp_visible(devs) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = visible.split(",") if visible else None
    return ",".join((phys[d.index].strip() if phys else str(d.index)) for d in devs)


def test_fuse_in_bit_identical(model_id, model_dir, devices):
    ref = run(model_dir, devices, "single", "default")
    again = run(model_dir, devices, "single", "default_again")
    if not all(torch.equal(ref[f"b{n}"], again[f"b{n}"]) for n in BATCHES):
        pytest.skip("default decode does not reproduce bitwise across processes on this model")
    got = run(model_dir, devices, "single", "fuse_in")
    for n in BATCHES:
        assert torch.equal(ref[f"b{n}"], got[f"b{n}"]), f"{n} rows: fused input rotation changed the logits"


@pytest.mark.parametrize("config", AGREE)
def test_switch_agrees(model_id, model_dir, devices, config):
    ref = run(model_dir, devices, "single", "default")
    got = run(model_dir, devices, "single", config)
    for n in BATCHES:
        assert_logits_agree(ref[f"b{n}"], got[f"b{n}"], ref["floor"], f"{config}, {n} rows")


@pytest.mark.tp
@pytest.mark.multi_gpu(2)
@pytest.mark.parametrize("config", ("ep_off", "sh_fold"))
def test_tp_expert_parallel_agrees(model_id, model_dir, devices, config):
    ref = run(model_dir, devices, "tp", "default")
    got = run(model_dir, devices, "tp", config)
    for n in BATCHES:
        assert_logits_agree(ref[f"b{n}"], got[f"b{n}"], ref["floor"], f"TP {config}, {n} rows")
