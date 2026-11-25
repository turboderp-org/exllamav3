import json
import math
from pathlib import Path

import pytest
import torch

from exllamav3.modules.factorized_attention import shift_kv
from exllamav3.modules.geodesic import geodesic_update
from testlib.parity import Gate, hf_cached, logits_floor_gate


@pytest.mark.nogpu
def test_shift_uses_raw_previous_kv_and_resets_at_start():
    current = torch.tensor([[[[2.0]], [[4.0]], [[8.0]]], [[[1.0]], [[3.0]], [[7.0]]]])
    last = torch.tensor([[[10.0]], [[9.0]]])
    positions = torch.tensor([[0, 1, 2], [8, 9, 10]])
    weights = torch.zeros(2, 3, 1)
    out = shift_kv(current, last, weights, positions)
    expected = torch.tensor([[2.0, 3.0, 6.0], [5.0, 2.0, 5.0]])
    torch.testing.assert_close(out.squeeze(-1).squeeze(-1), expected)
    split = shift_kv(current[:, 1:], current[:, 0], weights[:, 1:], positions[:, 1:])
    torch.testing.assert_close(split, out[:, 1:])


@pytest.mark.nogpu
def test_geodesic_update_preserves_radius_and_scales_angle_by_depth():
    residual = torch.tensor([[2.0, 0.0]])
    update = torch.tensor([[20.0, 2.0]])
    angle = math.pi / 8
    expected = torch.tensor([[2 * math.cos(angle), 2 * math.sin(angle)]])
    torch.testing.assert_close(geodesic_update(residual, update, 1.0, 0.0, 2), expected)
    torch.testing.assert_close(geodesic_update(residual, update * 1e24, 1.0, 0.0, 2), expected)


def test_olala_geodesic_fusion_preserves_fp32_arithmetic(device):
    rng = torch.Generator(device = device).manual_seed(83)
    residual = torch.randn((2, 17, 1536), generator = rng, device = device)
    update = torch.randn(residual.shape, generator = rng, device = device)
    compensation = torch.linspace(1, 9e10, 34, device = device).view(2, 17, 1)
    scale, bias = residual.new_tensor(0.75), residual.new_tensor(0.125)
    for decode in (False, True):
        expected = geodesic_update(residual, update * compensation, scale, bias, 19, use_triton = decode)
        actual = geodesic_update(residual, update, scale, bias, 19, compensation = compensation,
                                 preserve_reductions = not decode)
        torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


@pytest.mark.nogpu
@pytest.mark.model("olala-hf")
@pytest.mark.parametrize("multiplier", [None, 16])
def test_olala_latent_input_scaling(model_dir, tmp_path, multiplier):
    from exllamav3 import Config
    from exllamav3.modules.arch_specific.olala import OlalaLatentIn

    settings = json.loads((Path(model_dir) / "config.json").read_text())
    settings.pop("exl3_moe_activation_multiplier", None)
    if multiplier is not None:
        settings["exl3_moe_activation_multiplier"] = multiplier
    (tmp_path / "config.json").write_text(json.dumps(settings))
    config = Config.from_directory(str(tmp_path))
    layer = OlalaLatentIn(config, "down_proj", in_features = 128, out_features = 128)
    scaled, compensation = layer.scale_input(torch.ones(1, 128), {})
    expected = 128 if multiplier is None else multiplier
    torch.testing.assert_close(scaled, torch.full_like(scaled, 1 / expected))
    torch.testing.assert_close(compensation, torch.full_like(compensation, expected ** 2))


@pytest.mark.nogpu
def test_olala_latent_out_captures_subnormal_inputs_in_fp32(monkeypatch):
    from exllamav3.modules import Linear
    from exllamav3.modules.arch_specific.olala import OlalaLatentOut

    latent = OlalaLatentOut(config = None, key = "up_proj", in_features = 128, out_features = 128)
    subnormal = torch.full((1, 128), 2**-24, dtype = torch.half)

    def check_capture(layer, x, params):
        assert layer is latent
        torch.testing.assert_close(x, subnormal.float(), atol = 0, rtol = 0)

    monkeypatch.setattr(Linear, "capture_H", check_capture)
    latent.capture_H(subnormal, {})


@pytest.mark.nogpu
@pytest.mark.hf
@pytest.mark.model("olala-hf")
def test_olala_template(model_dir):
    from transformers import AutoTokenizer
    from exllamav3 import Config, Tokenizer
    from exllamav3.architecture.olala_template import render_olala_chat

    reference = AutoTokenizer.from_pretrained(model_dir, local_files_only = True)
    tokenizer = Tokenizer.from_config(Config.from_directory(model_dir))
    for turns in (
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "system", "content": "Réponds en français."}, {"role": "user", "content": "Quel jour sommes-nous ?"},
         {"role": "assistant", "content": "Je ne connais pas la date."}, {"role": "user", "content": "Merci."}],
    ):
        for effort in ("medium", "disabled"):
            expected = reference.apply_chat_template(
                turns, add_generation_prompt = True, reasoning_effort = effort, tokenize = False,
            )
            prompt = render_olala_chat(model_dir, turns, effort)
            assert prompt == expected
            got_ids = tokenizer.encode(prompt, add_bos = False, encode_special_tokens = True).flatten().tolist()
            assert got_ids == reference.encode(expected, add_special_tokens = False)
    assert tokenizer.eos_token_id == reference.eos_token_id


@pytest.mark.hf
@pytest.mark.slow
@pytest.mark.model("olala-hf")
def test_olala_checkpoint_logits(model_dir, device, monkeypatch):
    pytest.importorskip("einops")
    from transformers import AutoModelForCausalLM
    from testlib.parity import exl3_logits
    from exllamav3 import Config, Tokenizer

    monkeypatch.setenv("OLALA_TORCH_MAMBA3", "1")
    monkeypatch.setenv("OLALA_TORCH_MOE", "1")
    monkeypatch.setenv("OLALA_ATTN_IMPL", "eager")
    tokenizer = Tokenizer.from_config(Config.from_directory(model_dir))
    ids = tokenizer.encode("Le phare éclaire la mer. What is its purpose?", add_bos = False)

    @torch.inference_mode()
    def compute(dtype):
        reference = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype = dtype, trust_remote_code = True, local_files_only = True,
        ).eval()
        logits = reference(input_ids = ids, use_cache = False, logits_to_keep = 0).logits
        return {"logits": logits[0].float().cpu()}

    ref = hf_cached(model_dir, "olala-eager-f32", ids, lambda: compute(torch.float32))["logits"]
    floor = hf_cached(model_dir, "olala-eager-bf16", ids, lambda: compute(torch.bfloat16))["logits"]
    got = exl3_logits(model_dir, ids, device)
    gate = Gate("Olala logits vs reference")
    logits_floor_gate(gate, got, ref, floor)
    gate.assert_passes()


@pytest.mark.slow
@pytest.mark.model("olala-hf")
def test_olala_paged_state_and_prefix_reuse(model_dir, device):
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.generator.sampler import GreedySampler
    from testlib.parity import rfn

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4 * PAGE_SIZE, max_batch_size = 2)
    assert cache.num_layers == 7
    assert cache.num_recurrent_layers == 36
    model.load(device)
    generator = None
    try:
        tokenizer = Tokenizer.from_config(config)
        prefix = "The lighthouse illuminates the Atlantic at night. " * 30
        prompts = [prefix + "Explain why.", prefix + "Summarize this in French."]
        ids = tokenizer.encode(prompts[0], add_bos = False)[:, :PAGE_SIZE + 8]
        assert ids.shape[1] == PAGE_SIZE + 8
        with torch.inference_mode():
            state = cache.get_new_state()
            try:
                def forward(first, last, slot):
                    return model.forward(ids[:, first:last], {
                        "attn_mode": "flash_attn", "cache": cache,
                        "batch_shape": (1, 2 * PAGE_SIZE), "past_len": first,
                        "recurrent_states": [slot],
                    }).float().cpu()

                expected = forward(0, ids.shape[1], state)
                state.reset()
                prefill = forward(0, PAGE_SIZE, state)
                stashed = state.stash()
                other = cache.get_new_state()
                try:
                    other.position = PAGE_SIZE
                    other.unstash(stashed)
                    for layer in cache.get_all_recurrent_layers().values():
                        for tensor in layer.get_state_tensors():
                            assert torch.equal(tensor[state.slot], tensor[other.slot])
                    parts = [prefill, forward(PAGE_SIZE, PAGE_SIZE + 3, other)]
                    parts.extend(forward(i, i + 1, other) for i in range(PAGE_SIZE + 3, ids.shape[1]))
                    actual = torch.cat(parts, dim = 1)
                    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    # Different prefill lengths change FP16 projection and expert GEMM shapes.
                    assert rfn(actual[:, :PAGE_SIZE], expected[:, :PAGE_SIZE]) < 0.04
                    assert rfn(actual[:, PAGE_SIZE:], expected[:, PAGE_SIZE:]) < 0.03
                finally:
                    other.free()
            finally:
                state.free()

        generator = Generator(
            model, cache, tokenizer, max_batch_size = 2,
            recurrent_checkpoint_interval = PAGE_SIZE, max_chunk_size = PAGE_SIZE,
        )
        kwargs = dict(max_new_tokens = 5, completion_only = True, return_last_results = True)
        expected, _ = generator.generate(prompts[0], sampler = GreedySampler(), **kwargs)
        results, details = generator.generate(prompts, sampler = [GreedySampler(), GreedySampler()], **kwargs)
        assert results[0] == expected
        assert details[0]["cached_tokens"] >= PAGE_SIZE
        assert all(result["new_tokens"] == 5 for result in details)
    finally:
        if generator is not None:
            generator.close()
        model.unload()


@pytest.mark.nogpu
@pytest.mark.model("olala-hf")
def test_olala_router_retains_fp32_source_and_uses_unbiased_weights(model_dir):
    from exllamav3 import Config, Model

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    moe = model.modules[model.first_block_idx].mlp
    key = moe.routing_gate.key
    device = torch.device("cpu")
    moe.device = device
    moe.routing_gate.load(device)
    gate = config.stc.get_tensor(f"{key}.weight")
    bias = config.stc.get_tensor(f"{moe.key}.expert_bias")
    moe.e_score_correction_bias = bias
    moe.load_routing()
    torch.testing.assert_close(moe.routing_cfg.gate_tensor, gate.T)

    x = torch.randn(4, config.hidden_size, generator = torch.Generator().manual_seed(22)).half()
    from exllamav3.modules.block_sparse_mlp_routing import _routing_ds3_torch

    selected, weights = _routing_ds3_torch(x.shape[0], moe.routing_cfg, x, {})
    scores = (x.float() @ gate.T).sigmoid()
    expected = (scores + bias).topk(config.num_experts_per_tok).indices
    torch.testing.assert_close(selected.sort().values, expected.sort().values)
    unbiased = scores.gather(1, selected)
    expected_weights = config.expert_scale * unbiased / unbiased.sum(dim = -1, keepdim = True)
    torch.testing.assert_close(weights, expected_weights.half(), atol = 5e-4, rtol = 1e-3)


@pytest.mark.slow
@pytest.mark.model("olala-exl3")
def test_olala_exl3_prefill_and_decode(model_dir, device, monkeypatch):
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    from exllamav3.architecture.olala_template import render_olala_chat
    from exllamav3.generator.sampler import GreedySampler
    from exllamav3.modules import Linear
    from exllamav3.modules.arch_specific.olala_moe import pack_projections
    from exllamav3.util.backend import ROCM, gfx_arch

    monkeypatch.setenv("EXL3_OLALA_FUSED_MOE", "1")
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 1024, max_batch_size = 1)
    tokenizer = Tokenizer.from_config(config)
    model.load(device)
    generator = None
    try:
        assert any(isinstance(module, Linear) and module.quant_type == "exl3" for module in model)
        if not ROCM or gfx_arch(device) == "gfx1100":
            moe = model.modules[model.first_block_idx].mlp
            assert moe.decode_weights is not None
            # A premature read of a deferred trellis can produce finite but incorrect
            # projections. Reconstruct again after the checkpoint load has completed.
            dtype = torch.bfloat16 if moe.bf16_codebook is not None else torch.float16
            references = pack_projections(moe.decode_linears, dtype)
            for packed, reference in zip(moe.decode_weights, references, strict = True):
                torch.testing.assert_close(packed, reference, rtol = 0, atol = 0)
        recurrent_layers = cache.get_all_recurrent_layers()
        assert len(recurrent_layers) == config.num_hidden_layers
        for layer in recurrent_layers.values():
            tensors = layer.get_state_tensors()
            assert tensors and all(tensor.dtype == torch.float32 for tensor in tensors)
        prompt = render_olala_chat(
            model_dir, [{"role": "user", "content": "What is the capital of France? Answer with one word."}],
            "disabled",
        )
        ids = tokenizer.encode(prompt, add_bos = False, encode_special_tokens = True)
        # Exercise the multi-row EXL3 launches as well as the single-token decode path.
        for length in (16, 64):
            logits = model.forward(ids[:, :length])
            assert torch.isfinite(logits).all()

        generator = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = 256)
        channel_end = tokenizer.single_id("<|channel_end|>")
        answer, result = generator.generate(
            prompt, max_new_tokens = 32, sampler = GreedySampler(), completion_only = True,
            encode_special_tokens = True, decode_special_tokens = True,
            stop_conditions = [channel_end], return_last_results = True,
        )
        final = answer.partition("<|channel_start|>final<|content|>")[2].split("<|channel_end|>", 1)[0]
        assert final.strip().rstrip(".").lower() == "paris"
        assert result.get("eos_triggering_token_id") == channel_end, answer
    finally:
        if generator is not None:
            generator.close()
        model.unload()


@pytest.mark.rocm_only
@pytest.mark.slow
@pytest.mark.model("olala-exl3")
@pytest.mark.parametrize("batch_size, quant_cache", [(1, False), (2, False), (2, True)])
def test_olala_decode_graph_replays_slots_and_page_tables(model_dir, device, monkeypatch, batch_size, quant_cache):
    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.util.backend import gfx_arch

    if gfx_arch(device) != "gfx1100":
        pytest.skip("Olala decode graphs are enabled on gfx1100")
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    from exllamav3.cache import CacheLayer_quant
    cache_kwargs = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8) if quant_cache else {}
    eager_cache = Cache(model, max_num_tokens = 8 * PAGE_SIZE, max_batch_size = 2, **cache_kwargs)
    graph_cache = Cache(model, max_num_tokens = 8 * PAGE_SIZE, max_batch_size = 2, **cache_kwargs)
    model.load(device)
    prefixes = [
        tokenizer.encode(text * 50, add_bos = False)[:, :PAGE_SIZE + 4].contiguous()
        for text in ("A lighthouse on the coast. ", "Une forêt au bord de la mer. ")
    ]
    page_tables = (torch.tensor([[3, 1]], dtype = torch.int32), torch.tensor([[4, 6]], dtype = torch.int32))

    def run(cache):
        states = [cache.get_new_state() for _ in prefixes]

        def params(rows, positions):
            pages = max(positions) // PAGE_SIZE + 1
            return {
                "attn_mode": "flash_attn", "cache": cache,
                "cache_seqlens": torch.tensor(positions, dtype = torch.int32),
                "block_table": torch.cat([page_tables[i][:, :pages] for i in rows]),
                "recurrent_states": [states[i] for i in rows],
            }

        starts = [PAGE_SIZE - 2, PAGE_SIZE - 3]
        try:
            for i, ids in enumerate(prefixes):
                model.prefill(ids[:, :starts[i]], params([i], [0]))
            outputs = []
            for step in range(6):
                order = [1, 0] if step % 2 else [0, 1]
                for offset in range(0, len(order), batch_size):
                    rows = order[offset:offset + batch_size]
                    positions = [starts[i] + step for i in rows]
                    tokens = torch.cat([prefixes[i][:, pos:pos + 1] for i, pos in zip(rows, positions)])
                    outputs.append(model.forward(tokens, params(rows, positions)))
            assert [state.position for state in states] == [start + 6 for start in starts]
            return outputs
        finally:
            for state in states:
                state.free()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(model._decode_graph, "forward", lambda *args: None)
            expected = run(eager_cache)
        actual = run(graph_cache)
        assert len(model._decode_graph._entries) == 2
        assert all(entry.graph is not None for entry in model._decode_graph._entries.values())
        for got, ref in zip(actual, expected, strict = True):
            torch.testing.assert_close(got, ref, atol = 0.05, rtol = 0.02)
    finally:
        model.unload()
