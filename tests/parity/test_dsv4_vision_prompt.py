"""
DeepSeek-V4-Flash-Vision-Exp prompt-side checks on the stub (registry role dsv4-vision-stub), against the reference
code bundled with the checkpoint (inference/image_processor.py, inference/model.py's Gate.forward):

  1. image block alignment: the tokenizer must emit the block so that IMAGE_START lands on prompt position p with
     p % 4 == 3, i.e. exactly the reference image_processor's build_image_block(start_pos) layout for the block's
     actual start position
  2. bias_vl routing: chunks with image rows must route like the reference Gate.forward (image rows: top-k on
     scores + bias_vl, also in hash layers; text rows unchanged), and text-only chunks must keep the CUDA kernel
     path's selections
  3. exact chunking: the generator prefills each image span as exactly one nc chunk
"""

import os

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from testlib.parity import load_reference_module

pytestmark = pytest.mark.model("dsv4-vision-stub")


@pytest.fixture(scope = "module")
def stub(model_registry, device):
    from exllamav3 import Cache, Config, Model, Tokenizer
    model_dir = model_registry.get("dsv4-vision-stub").path
    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    vmodel = Model.from_config(config, component = "vision")
    vmodel.load(device)
    model = Model.from_config(config, component = "text")
    cache = Cache(model, max_num_tokens = 4096)      # before load: the cache allocates with the model
    model.load(device)
    model._test_cache = cache
    ref_ip = load_reference_module(os.path.join(model_dir, "inference", "image_processor.py"), "dsv4_image_processor")
    yield model_dir, config, tokenizer, vmodel, model, ref_ip
    model.unload()
    vmodel.unload()
    torch.cuda.empty_cache()


def _example_image(model_dir, name):
    return Image.open(os.path.join(model_dir, "inference", "examples", "images", name))


@torch.inference_mode()
def test_block_alignment(stub):
    model_dir, config, tokenizer, vmodel, model, ref_ip = stub
    from exllamav3.architecture.mm_processing.deepseek_v4 import IMAGE_END, IMAGE_PAD, IMAGE_START
    from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    mme = vmodel.get_image_embeddings(tokenizer, _example_image(model_dir, "corn.jpeg"))
    lh, lw = mme.metadata["grid_llm"]
    assert mme.align == 4 and mme.align_phase == 3 and mme.align_lead == 3
    full_types = mme.metadata["block_types"]
    assert full_types[:3] == [IMAGE_PAD] * 3 and full_types[3] == IMAGE_START and full_types[-1] == IMAGE_END

    seen_pads = set()
    for add_bos in (True, False):
        for prefix in ("", "a", "hello there", "one two three four", "The quick brown fox jumps", "x " * 9):
            ids = tokenizer.encode(prefix + mme.text_alias + " what is this?", add_bos = add_bos, embeddings = [mme])[0]
            mm = torch.nonzero(ids >= FIRST_MM_EMBEDDING_INDEX).flatten().tolist()
            assert mm, "no embedding ids emitted"
            start = mm[0]
            assert mm == list(range(start, start + len(mm))), "block must be contiguous"
            skip = int(ids[start]) - mme.first_index
            assert int(ids[mm[-1]]) == mme.last_index - 1, "block must end with IMAGE_END"
            emitted = full_types[skip:]
            # reference layout for this start position
            ref_types, _ = ref_ip.build_image_block(lh, lw, start)
            assert emitted == ref_types.tolist(), (start, skip)
            n_pad = emitted.index(IMAGE_START)
            assert (start + n_pad) % 4 == 3
            seen_pads.add(n_pad)
    assert seen_pads == {0, 1, 2, 3}, seen_pads


def _ref_gate(scores, bias, bias_vl, tid2eid, ids, image_mask, topk, route_scale):
    """Reference Gate.forward (inference/model.py) for score_func sqrtsoftplus."""
    if tid2eid is not None:
        indices = tid2eid[torch.where(image_mask, torch.zeros_like(ids), ids)]
        vl_indices = (scores + bias_vl).topk(topk, dim = -1)[1]
        indices = torch.where(image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices)
    else:
        b = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
        indices = (scores + b).topk(topk, dim = -1)[1]
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(dim = -1, keepdim = True) * route_scale
    return indices, weights


def _moe_layers(model):
    from exllamav3.modules import BlockSparseMLP
    out = []
    for m in model.modules:
        for sm in getattr(m, "modules", []):
            if isinstance(sm, BlockSparseMLP):
                out.append(sm)
    return out


@pytest.mark.parametrize("layer_idx", [0, 2, 3, 6])
@torch.inference_mode()
def test_bias_vl_routing(stub, layer_idx):
    model_dir, config, tokenizer, vmodel, model, ref_ip = stub
    from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    moe = _moe_layers(model)[layer_idx]
    cfg = moe.routing_cfg
    assert cfg.e_score_bias_vl is not None, "stub carries gate.bias_vl"
    is_hash = cfg.tid2eid is not None
    assert is_hash == (layer_idx < config.num_hash_layers)
    dev = moe.device
    torch.manual_seed(layer_idx)
    bsz = 48
    y = (torch.randn((bsz, config.hidden_size), device = dev) * 0.7).half()
    text_ids = torch.randint(0, config.vocab_size, (bsz,))
    image_mask = torch.zeros(bsz, dtype = torch.bool)
    image_mask[5:23] = True
    image_mask[40] = True
    mixed_ids = torch.where(image_mask, FIRST_MM_EMBEDDING_INDEX + torch.arange(bsz), text_ids)

    scores = F.softplus(torch.matmul(y.float(), cfg.gate_tensor.float())).sqrt()
    bias = cfg.e_score_correction_bias.float()
    bias_vl = cfg.e_score_bias_vl.float()
    ref_sel, ref_w = _ref_gate(scores, bias, bias_vl, cfg.tid2eid if is_hash else None,
                               mixed_ids.to(dev), image_mask.to(dev), cfg.num_experts_per_tok, cfg.routed_scaling_factor)

    # Mixed chunk: torch path with bias_vl
    params = {"input_ids": mixed_ids.unsqueeze(0), "indexed_embeddings": [object()]}
    sel, w = moe.routing_fn(bsz, cfg, y, params)
    assert torch.equal(sel.sort(dim = -1).values, ref_sel.sort(dim = -1).values), "expert sets differ from the reference gate"
    order = sel.argsort(dim = -1); ref_order = ref_sel.argsort(dim = -1)
    assert torch.allclose(w.float().gather(1, order), ref_w.gather(1, ref_order), rtol = 2e-2, atol = 1e-3)
    # image rows really used bias_vl: recompute with the text bias and expect a difference somewhere
    alt_sel, _ = _ref_gate(scores, bias, bias, cfg.tid2eid if is_hash else None,
                           mixed_ids.to(dev), image_mask.to(dev), cfg.num_experts_per_tok, cfg.routed_scaling_factor)
    assert not torch.equal(alt_sel.sort(dim = -1).values[image_mask.to(dev)], ref_sel.sort(dim = -1).values[image_mask.to(dev)]), \
        "bias_vl made no difference on this data; test is not discriminating"

    # Text-only chunk: kernel path, must agree with the reference on text semantics
    params = {"input_ids": text_ids.unsqueeze(0), "indexed_embeddings": [object()]}
    sel_t, w_t = moe.routing_fn(bsz, cfg, y, params)
    ref_sel_t, ref_w_t = _ref_gate(scores, bias, bias_vl, cfg.tid2eid if is_hash else None,
                                   text_ids.to(dev), torch.zeros(bsz, dtype = torch.bool, device = dev),
                                   cfg.num_experts_per_tok, cfg.routed_scaling_factor)
    same = (sel_t.sort(dim = -1).values == ref_sel_t.sort(dim = -1).values).all(dim = -1)
    assert same.float().mean() > 0.95, f"kernel path disagrees with the reference on {int((~same).sum())} of {bsz} rows"
    # No image rows -> the torch path must not have been taken (mask cached as None)
    assert params.get(("_vl_rows", str(dev))) is None


@torch.inference_mode()
def test_exact_chunking(stub):
    """The generator prefills each image span [IMAGE_START .. IMAGE_END] as exactly one chunk
    flagged nc_chunk, text chunks never contain span rows, nothing is re-fed, and decode
    continues afterwards. Covers a span crossing a page boundary, an image right after BOS,
    two adjacent images and a small chunk size."""
    model_dir, config, tokenizer, vmodel, model, ref_ip = stub
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler
    cache = model._test_cache
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 1, max_chunk_size = 512)
    images = [_example_image(model_dir, n) for n in ("corn.jpeg", "carrots.jpeg")]
    mmes = [vmodel.get_image_embeddings(tokenizer, im) for im in images]
    log = []
    orig = model.prefill
    def spy(input_ids, params = None):
        r = orig(input_ids, params)     # prepare_inputs sets nc_chunk on the params dict
        log.append((int(params["cache_seqlens"][0]), input_ids.shape[1], bool(params.get("nc_chunk", False))))
        return r
    model.prefill = spy
    try:
        long_text = " ".join(["word"] * 300)
        for label, prompt, mm in (
            ("text+img+text", "Describe " + mmes[0].text_alias + " briefly.", [mmes[0]]),
            ("page-crossing", long_text + " " + mmes[1].text_alias + " now", [mmes[1]]),
            ("img first", mmes[0].text_alias + " what?", [mmes[0]]),
            ("two adjacent", mmes[0].text_alias + mmes[1].text_alias + " compare", mmes),
        ):
            ids = tokenizer.encode(prompt, add_bos = True, embeddings = mm)
            job = Job(input_ids = ids, max_new_tokens = 4, sampler = ArgmaxSampler(), embeddings = mm)
            log.clear()
            gen.enqueue(job)
            final = None
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("eos"): final = r
            assert final is not None and final.get("new_tokens") == 4, (label, {k: v for k, v in (final or {}).items() if k != "job"}, log)
            # spans in this prompt
            row = ids[0]
            spans = []
            for e in mm:
                pos = torch.nonzero((row >= e.first_index + e.align_lead) & (row < e.last_index)).flatten()
                spans.append((int(pos[0]), int(pos[-1]) + 1))
            spans.sort()
            # chunks: contiguous, non-overlapping, covering the prompt minus its last token
            pos = 0
            for start, n, nc in log:
                assert start == pos, (label, log)
                pos += n
            assert pos == ids.shape[1] - 1, (label, pos, ids.shape, log)
            nc_chunks = [(s, s + n) for s, n, nc in log if nc]
            assert nc_chunks == spans, (label, nc_chunks, spans, log)
            for s, n, nc in log:
                if nc:
                    continue
                for a, b in spans:
                    assert s + n <= a or s >= b, (label, "text chunk overlaps a span", log)
    finally:
        model.prefill = orig
