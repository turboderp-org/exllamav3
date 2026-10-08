"""
Kimi Linear (KimiLinearForCausalLM) per-stage parity against transformers' native implementation (>= 5.17; the
checkpoint's bundled remote code targets 4.57 and weights experts by bias-shifted scores, unlike the native class,
vLLM and DeepSeek-V3's reference) on real weights (registry role kimi-linear-hf, unquantized). The 48B checkpoint is
96 GB in bf16, so the HF side is spread over two devices with accelerate's planner while exllamav3 streams one
module at a time on the test device. HF references are disk-cached ($EXL3_HFREF_CACHE).

Cascaded: exllamav3 states after the embedding, every block but the last and the final norm against HF bf16 eager,
judged against the HF eager-vs-sdpa floor (testlib.parity.FLOOR_K). Isolated: every exllamav3 block is fed HF's own
input to that block (teacher forcing) against an fp16 reference (no bf16-vs-fp16 routing flips), so each block's
error is measured on its own instead of accumulating through the routing cascade; the report attributes each MoE
block's error to the tokens whose expert set differs from fp32 routing on the same input and to those that agree.
"""

import pytest
import torch

from testlib.parity import (Gate, LIGHTHOUSE_TEXT, exl3_stream, hf_aligned_capture, hf_device_map, hf_reference,
                            hidden_state_gate, logits_floor_gate, rfn)

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.multi_gpu(2), pytest.mark.model("kimi-linear-hf")]

SEQ_LEN = 512
# Free memory left on the two HF devices: the first takes the conversion transients of the fused expert tensors
HF_HEADROOM_GIB = [16.0, 6.0]


@pytest.fixture(scope = "module")
def input_ids(model_registry):
    from exllamav3 import Config, Tokenizer
    model_dir = model_registry.get("kimi-linear-hf").path
    return Tokenizer.from_config(Config.from_directory(model_dir)).encode(LIGHTHOUSE_TEXT)[:, :SEQ_LEN]


def _hf(model_dir, ids, devices, attn_impl, dtype):
    from transformers import AutoModelForCausalLM
    return hf_reference(AutoModelForCausalLM, model_dir, ids, None, variant = "native", dtype = dtype,
                        attn_impl = attn_impl, experts_impl = "eager",
                        device_map = hf_device_map(model_dir, devices[:2], dtype, headroom_gib = HF_HEADROOM_GIB))


def _routing_flip_hooks(flips: dict):
    """configure() for exl3_stream: record, per MoE block, the tokens whose selected expert set differs from fp32
    routing (sigmoid scores + correction bias, top-k) on the same input"""
    def configure(model):
        from exllamav3.modules import TransformerBlock
        for block in model.modules:
            mlp = getattr(block, "mlp", None) if isinstance(block, TransformerBlock) else None
            if mlp is None or not hasattr(mlp, "routing_fn"):
                continue

            def hooked(bsz, cfg, z, p, orig = mlp.routing_fn, layer_idx = block.layer_idx):
                sel, w = orig(bsz, cfg, z, p)
                scores = (z.float() @ cfg.gate_tensor.float()).sigmoid()
                if cfg.e_score_correction_bias is not None:
                    scores = scores + cfg.e_score_correction_bias.float()
                ref = scores.topk(sel.shape[-1], dim = -1).indices
                flips[layer_idx] = (sel.long().sort(dim = -1).values != ref.sort(dim = -1).values).any(dim = -1).cpu()
                return sel, w

            mlp.routing_fn = hooked
    return configure


def test_cascaded(model_dir, input_ids, device, devices):
    ref = _hf(model_dir, input_ids, devices, "eager", torch.bfloat16)
    floor = _hf(model_dir, input_ids, devices, "sdpa", torch.bfloat16)
    states, logits, _ = exl3_stream(model_dir, input_ids, device, hf_aligned_capture(embed_after = 0))
    gate = Gate("Kimi Linear vs HF bf16 eager, floor HF sdpa")
    hidden_state_gate(gate, states, ref["hs"], floor["hs"])
    logits_floor_gate(gate, logits, ref["logits"], floor["logits"])
    gate.assert_passes()


def test_isolated_blocks(model_dir, input_ids, device, devices):
    from exllamav3.modules import TransformerBlock
    ref = _hf(model_dir, input_ids, devices, "eager", torch.float16)
    floor = _hf(model_dir, input_ids, devices, "sdpa", torch.float16)
    hs = ref["hs"]

    def feed(model, idx, module, state, states):
        if isinstance(module, TransformerBlock):
            return hs[len(states) - 1].to(state.device, state.dtype)
        return state

    flips = {}
    states, logits, _ = exl3_stream(model_dir, input_ids, device, hf_aligned_capture(embed_after = 0),
                                    configure = _routing_flip_hooks(flips), feed = feed)
    gate = Gate("Kimi Linear, teacher-forced blocks vs HF fp16 eager, floor HF sdpa")
    hidden_state_gate(gate, states, hs, floor["hs"])
    logits_floor_gate(gate, logits, ref["logits"], floor["logits"])
    for i in sorted(flips):
        a = states[i + 1].float().flatten(0, -2)
        b = hs[i + 1].float().flatten(0, -2)
        m = flips[i]
        if m.shape[0] != a.shape[0]:
            continue
        kept = rfn(a[~m], b[~m]) if (~m).any() else float("nan")
        flipped = rfn(a[m], b[m]) if m.any() else float("nan")
        gate.info(f"layer {i} routing", f"{100.0 * m.float().mean().item():.2f}% flipped, rfn kept {kept:.6f}, "
                                        f"flipped {flipped:.6f}")
    gate.assert_passes()
