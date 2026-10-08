"""
DeepSeek-V4-Flash-Vision-Exp vision tower parity against the reference implementation bundled with the checkpoint
(inference/vision.py + inference/image_processor.py), on the stub (registry role dsv4-vision-stub), per image:

  1. preprocessing: the resize schedule / grids must match exactly, pixel patches to fp16 rounding
  2. tower + aligner on the SAME patches: exllamav3 (fp16) vs the reference in fp32, within twice the reference's own
     bf16-vs-fp32 distance (the calibration bar)
  3. end to end: exllamav3's get_image_embeddings block (own preprocessing) vs the reference block assembled from
     the fp32 reference embeddings of exllamav3's own fp32 pixels and image_processor.build_image_block

The odd-sized synthetic images exercise the resize schedule, gray padding and the aligner's edge padding.
"""

import glob
import io
import json
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from testlib.parity import Gate, load_reference_module, make_test_image, min_cos, rfn

pytestmark = pytest.mark.model("dsv4-vision-stub")

IMAGES = ["synthetic", "carrots.jpeg", "corn.jpeg", "wide", "tiny"]


def load_ref_weights(model_dir: str, prefixes: tuple) -> dict:
    """Tensors whose names start with any prefix, from all shards (header scan)"""
    from safetensors import safe_open
    sd = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        keys = [k for k in hdr if k != "__metadata__" and k.startswith(prefixes)]
        if keys:
            with safe_open(f, "pt") as sf:
                for k in keys:
                    sd[k] = sf.get_tensor(k)
    return sd


@pytest.fixture(scope = "module")
def stub(model_registry, device):
    """(model dir, reference vision module, reference image processor, reference args, {"bf16"|"fp32": (vit,
    aligner)}, exl3 vision model, tokenizer or None)"""
    from exllamav3 import Config, Model, Tokenizer
    model_dir = model_registry.get("dsv4-vision-stub").path
    inf_dir = os.path.join(model_dir, "inference")
    ref_vision = load_reference_module(os.path.join(inf_dir, "vision.py"), "dsv4_vision")
    ref_ip = load_reference_module(os.path.join(inf_dir, "image_processor.py"), "dsv4_image_processor")
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    rargs = SimpleNamespace(dim = cfg["hidden_size"], **{k: v for k, v in cfg.items() if k.startswith("vision_")})

    sd = load_ref_weights(model_dir, ("vision.", "aligner."))
    vit_sd = {k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")}
    al_sd = {k[len("aligner."):]: v for k, v in sd.items() if k.startswith("aligner.")}
    refs = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        vit = ref_vision.ViT(rargs)
        vit.load_state_dict(vit_sd, strict = True)
        al = ref_vision.Aligner(rargs)
        al.load_state_dict(al_sd, strict = True)
        refs[name] = (vit.to(device, dtype).eval(), al.to(device, dtype).eval())

    config = Config.from_directory(model_dir)
    assert "vision" in config.model_classes, "vision component not registered"
    vmodel = Model.from_config(config, component = "vision")
    vmodel.load(device)
    try:
        tokenizer = Tokenizer.from_config(config)
    except Exception:
        tokenizer = None    # donor dirs (convert_vision output) carry no tokenizer; unused by the tower
    markers = load_ref_weights(model_dir, ("image_",))
    yield SimpleNamespace(model_dir = model_dir, ip = ref_ip, args = rargs, refs = refs, vmodel = vmodel,
                          tokenizer = tokenizer, markers = markers)
    vmodel.unload()
    del refs
    torch.cuda.empty_cache()


def _image(stub, name):
    from PIL import Image
    if name == "synthetic":
        return make_test_image()
    if name == "wide":
        return make_test_image(1600, 180)
    if name == "tiny":
        return make_test_image(90, 70)
    path = os.path.join(stub.model_dir, "inference", "examples", "images", name)
    if not os.path.exists(path):
        pytest.skip(f"stub has no example image {name}")
    return Image.open(path)


@torch.inference_mode()
def _ref_tower(stub, name, patches, h, w, device):
    vit, al = stub.refs[name]
    dtype = torch.bfloat16 if name == "bf16" else torch.float32
    with torch.device(device):      # the reference builds its rope tables on the default device
        return al(vit(patches.to(device, dtype), h, w), h, w).float()


@pytest.mark.parametrize("image_name", IMAGES)
def test_tower_parity(stub, image_name, device):
    image = _image(stub, image_name)
    gate = Gate(f"DeepSeek-V4 vision, {image_name} {image.size[0]}x{image.size[1]}")
    rp_w = stub.args.vision_patch_size

    # 1) preprocessing
    buf = io.BytesIO()
    image.save(buf, format = "PNG")
    rp, r_h, r_w, r_lh, r_lw = stub.ip.load_image({"data": buf.getvalue()}, stub.args)     # (N, 3, p, p) bf16
    ep, e_h, e_w, e_lh, e_lw = stub.vmodel.preprocess(image)                              # (N, 3 p p) half
    gate.true("grids", (r_h, r_w, r_lh, r_lw) == (e_h, e_w, e_lh, e_lw),
              f"ref vit {r_h}x{r_w} llm {r_lh}x{r_lw}, exl3 vit {e_h}x{e_w} llm {e_lh}x{e_lw}")
    rp_flat = rp.reshape(rp.shape[0], -1)
    if ep.shape == rp_flat.shape:
        gate.lt("pixels rfn", rfn(ep, rp_flat), 5e-3)
    else:
        gate.true("pixel patches shape", False, f"{tuple(ep.shape)} vs {tuple(rp_flat.shape)}")
        gate.assert_passes()

    # 2) tower + aligner on the reference patches (bf16 values are exact in fp16)
    out32 = _ref_tower(stub, "fp32", rp, r_h, r_w, device)
    bar = rfn(_ref_tower(stub, "bf16", rp, r_h, r_w, device), out32)
    with torch.inference_mode():
        out_e = stub.vmodel.encode_patches(rp_flat.to(device).half(), r_h, r_w).float()
    gate.true("aligner output shape", out_e.shape == out32.shape, f"{tuple(out_e.shape)} vs {tuple(out32.shape)}")
    gate.lt("tower rfn vs fp32", rfn(out_e, out32), max(2.0 * bar, 1e-3))
    gate.info("tower", f"bf16-vs-fp32 bar {bar:.2e}, min cos vs fp32 {min_cos(out_e, out32):.5f}")

    # 3) end-to-end block: exllamav3 (own fp16 pixels, fp16 tower) vs the fp32 reference on exllamav3's own fp32
    #    pixels, laid out with the reference block builder
    with torch.inference_mode():
        mme = stub.vmodel.get_image_embeddings(stub.tokenizer, image)
    ep32 = stub.vmodel.preprocess(image, dtype = torch.float)[0]
    ref32 = _ref_tower(stub, "fp32", ep32.view(-1, 3, rp_w, rp_w), e_h, e_w, device).cpu()
    types, perm = stub.ip.build_image_block(e_lh, e_lw, 3)
    markers = torch.stack([stub.markers[k] for k in
                           ("image_start", "image_pad", "image_pad", "image_newline", "image_end")]).float()
    block = markers[types].clone()
    img_rows = types == stub.ip.IMAGE
    block[img_rows] = ref32[perm]
    e_block = mme.embeddings[mme.align_lead:].float()      # drop the spare leading pads (reference start_pos 3: none)
    gate.true("block rows", e_block.shape[0] == block.shape[0], f"{e_block.shape[0]} vs reference {block.shape[0]}")
    if e_block.shape[0] == block.shape[0]:
        gate.lt("image rows rfn", rfn(e_block[img_rows], block[img_rows]), max(2.0 * bar, 2e-3))
        gate.lt("marker rows max abs", (e_block[~img_rows] - block[~img_rows]).abs().max().item(), 1e-3)
    gate.assert_passes()
