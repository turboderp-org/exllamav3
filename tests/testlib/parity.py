"""
Shared machinery for the per-architecture parity tests (tests/parity/): exllamav3 against HF transformers or a
checkpoint's bundled reference code. A test file declares the checkpoint (tiny builder or registry role), the reference
class / loader, how exllamav3 states align with the reference's, and its thresholds; everything else lives here.

    metrics     rfn, min_cos, tensor_stats (hidden states), logit_stats (KL, argmax, margin-gated argmax, top-5)
    Gate        named threshold checks collected into one report: assert_passes() for the parity test, assert_fails()
                for the mutation test proving the same gate catches a broken model ($EXL3_PARITY_REPORT=1 with -s
                prints every report)
    policies    floor_check (within FLOOR_K x a reference noise floor), control_check (within k x a lower-precision
                run of the reference), hidden_state_gate / logits_floor_gate (per-stage and logit floor policy),
                logit_gate (absolute KL / argmax thresholds)
    HF          load_hf (dtype, attn_implementation, experts_implementation, fp8 dequant, device or device_map),
                hf_device_map (spread over devices), hf_forward (logits + hidden states), hf_cached / hf_reference
                (disk cache under $EXL3_HFREF_CACHE), hf_vision_tower (a vision tower class from a stub's tensors)
    exl3        exl3_logits (whole model, cache-less), exl3_stream (one module resident at a time, with capture and
                teacher-forcing hooks), hf_aligned_capture (the usual alignment with HF's hidden_states)
    inputs      token_ids, hf_tokenizer_ids, RUMEN_TEXT / LIGHTHOUSE_TEXT, make_test_image
    checkpoints truncated_checkpoint (config-edited symlink farm), trim_layer_lists, load_reference_module
    mutation    mutate / mutated (set an attribute on every matching object, permanently or for a context)

    ref = hf_reference(Cls, model_dir, ids, device, dtype = torch.bfloat16, attn_impl = "eager")
    floor = hf_reference(Cls, model_dir, ids, device, dtype = torch.bfloat16, attn_impl = "sdpa")
    states, logits, _ = exl3_stream(model_dir, ids, device, hf_aligned_capture(embed_after = 0))
    gate = Gate("stub vs HF")
    hidden_state_gate(gate, states, ref["hs"], floor["hs"])
    logits_floor_gate(gate, logits, ref["logits"], floor["logits"])
    gate.assert_passes()
"""

import contextlib
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


# Metrics

def rfn(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative Frobenius norm of the difference, ||got - ref|| / ||ref||, fp32"""
    got, ref = got.float(), ref.float()
    return ((got - ref).norm() / (ref.norm() + 1e-12)).item()


def min_cos(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Smallest cosine similarity over rows (last dimension is the feature dimension)"""
    a = got.float().reshape(-1, got.shape[-1])
    b = ref.float().reshape(-1, ref.shape[-1])
    return F.cosine_similarity(a, b, dim = -1).min().item()


@dataclass
class TensorStats:
    max_abs: float
    rfn: float
    min_cos: float

    def __str__(self):
        return f"max_abs {self.max_abs:.5f} rfn {self.rfn:.6f} min_cos {self.min_cos:.6f}"


def tensor_stats(got: torch.Tensor, ref: torch.Tensor) -> TensorStats:
    assert got.shape == ref.shape, f"shape {tuple(got.shape)} vs reference {tuple(ref.shape)}"
    return TensorStats((got.float() - ref.float()).abs().max().item(), rfn(got, ref), min_cos(got, ref))


@dataclass
class LogitStats:
    kl: torch.Tensor            # per-row KL(ref || got), float64
    argmax: float               # top-1 agreement over all rows
    argmax_conf: float          # top-1 agreement over rows where the reference's top-1 margin exceeds `margin`
    top5: float                 # mean top-5 set overlap
    max_abs: float

    @property
    def kl_mean(self) -> float:
        return self.kl.mean().item()

    @property
    def kl_max(self) -> float:
        return self.kl.max().item()

    def __str__(self):
        return (f"KL mean {self.kl_mean:.3e} max {self.kl_max:.3e}, argmax {self.argmax * 100:.2f}% "
                f"(confident {self.argmax_conf * 100:.2f}%), top-5 {self.top5 * 100:.1f}%, max_abs {self.max_abs:.4f}")


def logit_stats(got: torch.Tensor, ref: torch.Tensor, margin: float = 0.5) -> LogitStats:
    """Statistics of (rows, vocab) logits against reference logits, in float64"""
    got = got.reshape(-1, got.shape[-1]).double()
    ref = ref.reshape(-1, ref.shape[-1]).double()
    assert got.shape == ref.shape, f"logits {tuple(got.shape)} vs reference {tuple(ref.shape)}"
    lp_r = torch.log_softmax(ref, dim = -1)
    lp_g = torch.log_softmax(got, dim = -1)
    kl = (lp_r.exp() * (lp_r - lp_g)).sum(-1)
    agree = ref.argmax(-1) == got.argmax(-1)
    top2 = ref.topk(2, dim = -1).values
    conf = (top2[:, 0] - top2[:, 1]) > margin
    argmax_conf = agree[conf].float().mean().item() if conf.any() else 1.0
    t5r = ref.topk(5, dim = -1).indices
    t5g = got.topk(5, dim = -1).indices
    top5 = (t5r.unsqueeze(-1) == t5g.unsqueeze(-2)).any(-1).float().mean().item()
    return LogitStats(kl, agree.float().mean().item(), argmax_conf, top5, (got - ref).abs().max().item())


# Gate

class Gate:
    """A set of named threshold checks evaluated together, so a failure reports every metric, and so the same
    gate can be asserted to pass (parity) or to fail (mutation test)"""

    def __init__(self, tag: str = ""):
        self.tag = tag
        self.lines = []
        self.failures = []

    def _add(self, name: str, ok: bool, desc: str):
        self.lines.append(f"  {'ok  ' if ok else 'FAIL'} {name}: {desc}")
        if not ok:
            self.failures.append(name)
        return ok

    def lt(self, name: str, value: float, bound: float):
        return self._add(name, value < bound, f"{value:.4e} < {bound:.4e}")

    def le(self, name: str, value: float, bound: float):
        return self._add(name, value <= bound, f"{value:.4e} <= {bound:.4e}")

    def gt(self, name: str, value: float, bound: float):
        return self._add(name, value > bound, f"{value:.6f} > {bound:.6f}")

    def ge(self, name: str, value: float, bound: float):
        return self._add(name, value >= bound, f"{value:.6f} >= {bound:.6f}")

    def true(self, name: str, cond: bool, detail: str = ""):
        return self._add(name, bool(cond), detail)

    def info(self, name: str, detail: str):
        self.lines.append(f"       {name}: {detail}")

    @property
    def passed(self) -> bool:
        return not self.failures

    def report(self) -> str:
        return "\n".join([f"{self.tag}:"] + self.lines)

    def _show(self):
        # $EXL3_PARITY_REPORT=1 (with -s) prints every gate's metrics, also when it passes
        if os.environ.get("EXL3_PARITY_REPORT"):
            print("\n" + self.report())

    def assert_passes(self):
        self._show()
        assert self.passed, f"parity gate failed ({', '.join(self.failures)})\n{self.report()}"

    def assert_fails(self):
        self._show()
        assert not self.passed, f"mutated model passed the parity gate (mutation not detected)\n{self.report()}"


# Tolerance policies

# exllamav3 (fp16 pipeline) against an HF reference may sit this far above the reference's own noise floor (HF
# eager vs sdpa, or bf16 vs fp32): a correct implementation lands at or below the floor, structural errors land
# several times above it
FLOOR_K = 2.0


def floor_check(gate: Gate, name: str, err: float, floor: float, k: float = FLOOR_K):
    """err <= k x floor"""
    return gate._add(name, err <= k * floor, f"{err:.4e} <= {k:g} x floor {floor:.4e}")


def control_check(gate: Gate, name: str, err: float, control: float, k: float = 3.0, eps: float = 1e-6):
    """err < k x control + eps: exllamav3's distance from a high-precision reference is within k times that of a
    lower-precision run of the reference itself (the precision the model is shipped in)"""
    return gate._add(name, err < k * control + eps, f"{err:.4e} < {k:g} x control {control:.4e}")


def hidden_state_gate(gate: Gate, got: list, ref: list, floor: list | None = None, first: int = 1,
                      k: float = FLOOR_K, names: list[str] | None = None):
    """Per-stage parity of captured hidden states against the reference's, from index `first` on: the worst stage
    rfn within k x the worst rfn of a floor run (another configuration of the reference) on the same stages"""
    gate.true("stage count", len(got) == len(ref), f"{len(got)} captured vs {len(ref)} reference states")
    if len(got) != len(ref):
        return
    names = names or [f"state {i}" for i in range(len(ref))]
    worst = worst_floor = 0.0
    for i in range(first, len(ref)):
        st = tensor_stats(got[i].view_as(ref[i]), ref[i])
        desc = str(st)
        worst = max(worst, st.rfn)
        if floor is not None:
            f = rfn(floor[i], ref[i])
            worst_floor = max(worst_floor, f)
            desc += f" (floor rfn {f:.6f})"
        gate.info(names[i], desc)
    if floor is not None:
        floor_check(gate, "worst stage rfn", worst, worst_floor, k)


def logits_floor_gate(gate: Gate, got: torch.Tensor, ref: torch.Tensor, floor: torch.Tensor, k: float = FLOOR_K):
    """Logits rfn within k x the floor run's, with KL / top-1 agreement reported"""
    floor_check(gate, "logits rfn", rfn(got, ref), rfn(floor, ref), k)
    gate.info("logits", str(logit_stats(got, ref)))
    gate.info("floor logits", str(logit_stats(floor, ref)))


def logit_gate(gate: Gate, st: LogitStats, prefix: str = "", kl_mean: float | None = None,
               kl_max: float | None = None, argmax: float | None = None, argmax_conf: float | None = None):
    """The usual logit thresholds: KL mean/max below, argmax (all rows / confident rows) above"""
    p = f"{prefix} " if prefix else ""
    if kl_mean is not None:
        gate.lt(f"{p}KL mean", st.kl_mean, kl_mean)
    if kl_max is not None:
        gate.lt(f"{p}KL max", st.kl_max, kl_max)
    if argmax is not None:
        gate.gt(f"{p}argmax", st.argmax, argmax)
    if argmax_conf is not None:
        gate.gt(f"{p}argmax (confident)", st.argmax_conf, argmax_conf)
    gate.info(f"{p}logits", str(st))


# HF side

def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def hf_config_quant_method(model_dir: str) -> str | None:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return (cfg.get("quantization_config") or {}).get("quant_method")


def load_hf(cls, model_dir: str, device = None, dtype: torch.dtype = torch.float32, attn_impl: str = "eager",
            experts_impl: str | None = None, device_map = None, **kwargs):
    """from_pretrained with the reference settings: dtype, attention implementation, expert implementation
    ("eager" runs everywhere; grouped_mm needs SM90 and one device). FP8 block-scaled checkpoints are dequantized
    to dtype, so both sides compute from the same weights. Whole model on `device`, or `device_map`"""
    if experts_impl is not None:
        kwargs["experts_implementation"] = experts_impl
    if hf_config_quant_method(model_dir) == "fp8" and "quantization_config" not in kwargs:
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize = True)
    if device_map is not None:
        kwargs["device_map"] = device_map
    elif device is not None:
        kwargs["device_map"] = str(device)
    model = cls.from_pretrained(model_dir, dtype = dtype, attn_implementation = attn_impl, **kwargs)
    if "quantization_config" in kwargs and dtype == torch.float32:
        # Dequantized weights come out in the quantizer's working dtype (bf16); upcasting is lossless. Never cast
        # down: fp32 buffers such as router score biases must keep their precision
        model = model.to(dtype)
    return model.eval()


def hf_device_map(model_dir: str, devices: list, dtype: torch.dtype, headroom_gib: float | list[float] = 20.0,
                  trust_remote_code: bool = False) -> dict:
    """Per-layer device map spreading an HF model over `devices` (accelerate's planner), leaving headroom_gib (one
    value, or one per device) free for activations and transformers' conversion transients (fused expert tensors)"""
    from exllamav3.util.hf_util import hf_device_map_from_split
    if not isinstance(headroom_gib, (list, tuple)):
        headroom_gib = [headroom_gib] * len(devices)
    split = [0.0] * torch.cuda.device_count()
    for d, h in zip(devices, headroom_gib):
        split[d.index] = torch.cuda.get_device_properties(d).total_memory / 2**30 - h
    return hf_device_map_from_split(model_dir, split, dtype, trust_remote_code = trust_remote_code)


@torch.inference_mode()
def hf_forward(model, input_ids: torch.Tensor, hidden_states: bool = True, **kwargs) -> dict:
    """{"logits": (seq, vocab) fp32 CPU, "hs": [hidden states on the CPU]} of a batch-1 forward"""
    dev = model.get_input_embeddings().weight.device
    out = model(input_ids = input_ids.to(dev), use_cache = False, output_hidden_states = hidden_states, **kwargs)
    res = {"logits": out.logits[0].float().cpu()}
    if hidden_states:
        res["hs"] = [h.cpu() for h in out.hidden_states]
    return res


def hfref_dir() -> str:
    return os.environ.get("EXL3_HFREF_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "exl3_hfref"))


def hf_cached(model_dir: str, variant: str, input_ids: torch.Tensor, compute: Callable[[], dict]) -> dict:
    """Disk-cached reference outputs: an eager many-expert HF forward over a real-weight stub takes minutes and never
    changes for a given (checkpoint, input, variant). The key covers the checkpoint path and config.json, the
    variant string (attn impl, dtype, overrides) and the exact input ids"""
    h = hashlib.sha256()
    # The weights' real location, so a symlink farm with an edited config shares the source's cache entries
    shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    src = os.path.dirname(os.path.realpath(os.path.join(model_dir, shards[0]))) if shards else os.path.abspath(model_dir)
    h.update(src.encode())
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        h.update(f.read())
    h.update(variant.encode())
    h.update(input_ids.cpu().to(torch.int64).numpy().tobytes())
    name = f"hfref_{os.path.basename(src.rstrip('/'))}_{variant}_{h.hexdigest()[:16]}.pt"
    path = os.path.join(hfref_dir(), name)
    if os.path.exists(path):
        return torch.load(path, weights_only = True)
    res = compute()
    os.makedirs(hfref_dir(), exist_ok = True)
    torch.save(res, path + ".tmp")
    os.replace(path + ".tmp", path)
    return res


def hf_reference(cls, model_dir: str, input_ids: torch.Tensor, device, variant: str = "", **load_kwargs) -> dict:
    """hf_cached(hf_forward(load_hf(...))): loads, runs and frees the HF model unless the result is cached.
    variant names any setting not captured by load_kwargs (e.g. a config override)"""
    key = "_".join([variant] + [f"{k}-{v}" for k, v in sorted(load_kwargs.items()) if k not in ("config", "device_map")]
                   ).replace("torch.", "").strip("_")

    def compute():
        model = load_hf(cls, model_dir, device = device, **load_kwargs)
        try:
            return hf_forward(model, input_ids)
        finally:
            del model
            free_cuda()

    return hf_cached(model_dir, key, input_ids, compute)


# exllamav3 side

@torch.inference_mode()
def exl3_logits(model_dir: str, input_ids: torch.Tensor, device, configure: Callable | None = None) -> torch.Tensor:
    """Cache-less full-sequence logits (seq, vocab) fp32 on the CPU from a whole-model load. configure(model) runs
    after loading (mutations, overrides)"""
    from exllamav3 import Config, Model
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(torch.device(device))
    try:
        if configure is not None:
            configure(model)
        return model.forward(input_ids.cpu(), {"attn_mode": "flash_attn_nc"})[0].float().cpu()
    finally:
        model.unload()
        free_cuda()


def hf_aligned_capture(embed_after: int = 0) -> Callable:
    """Capture hook for exl3_stream matching HF's hidden_states tuple [embeddings, layer 0 out, ..., layer N-2 out,
    norm(layer N-1 out)]: the state after module `embed_after` (the embedding, or the stream expansion of
    hyper-connection models, whose HF hidden states are the stream stacks), every block output except the last,
    and the final norm's output in place of the last block's"""
    def capture(model, idx, module, state):
        from exllamav3.modules import RMSNorm, TransformerBlock
        if idx == embed_after:
            return state
        if isinstance(module, TransformerBlock) and idx != model.last_kv_module_idx:
            return state
        if isinstance(module, RMSNorm):
            return state
        return None
    return capture


@torch.inference_mode()
def exl3_stream(model_dir: str, input_ids: torch.Tensor, device, capture: Callable, configure: Callable | None = None,
                feed: Callable | None = None, params: dict | None = None):
    """Forward through the model with one module resident at a time (stub checkpoints with real-sized layers fit on
    any device). capture(model, idx, module, state) returns a tensor to record after the module or None;
    feed(model, idx, module, state, states) may replace a module's input (teacher forcing). configure(model) runs
    before the loop. Returns (captured states on the CPU, logits (seq, vocab) fp32 CPU, model)"""
    from exllamav3 import Config, Model
    from exllamav3.util.memory import free_mem
    config = Config.from_directory(model_dir)
    config.override_dynamic_seq_len(input_ids.shape[1])
    model = Model.from_config(config)
    if configure is not None:
        configure(model)
    params = {} if params is None else params
    dev = torch.device(device)
    state = model.prepare_inputs(input_ids, params)
    states = []
    logits = None
    for idx, module in enumerate(model.modules):
        config.stc.begin_deferred_load()
        module.load(dev)
        config.stc.end_deferred_load()
        state = module.prepare_for_device(state, params)
        if feed is not None:
            state = feed(model, idx, module, state, states)
        state = module.forward(state, params)
        c = capture(model, idx, module, state)
        if c is not None:
            states.append(c.cpu())
        if idx == model.logit_layer_idx:
            logits = state[0].float().cpu()
        module.unload()
        config.stc.close()
        free_mem()
    return states, logits, model


def hf_vision_tower(vision_cls, hf_config, model_dir: str, prefix: str, device, dtype: torch.dtype = torch.bfloat16):
    """An HF vision tower class built from hf_config.vision_config with the checkpoint's `prefix` tensors (stubs carry
    the tower next to a truncated text model, so the full model class is never instantiated)"""
    from safetensors import safe_open
    model = vision_cls._from_config(hf_config.vision_config, torch_dtype = dtype)
    sd = {}
    for fn in sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(model_dir, fn), framework = "pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = f.get_tensor(k)
    missing, _ = model.load_state_dict(sd, strict = False)
    assert not missing, f"vision tensors missing from the checkpoint: {missing[:5]}"
    return model.to(device).eval()


# Misc

# Plain English reference texts (repeated so any tokenizer yields several hundred tokens)
RUMEN_TEXT = (
    "The digestive system of the cow allows it to survive on a diet of grasses and other "
    "fibrous plants that many other mammals could not process. Its stomach is divided into "
    "four compartments, the largest of which, the rumen, hosts a dense community of microbes "
    "that ferment cellulose into fatty acids the animal can absorb. In exchange the microbes "
    "receive a warm, stable environment and a steady supply of food. This arrangement, refined "
    "over millions of years, is one of the most studied examples of symbiosis in agriculture, "
    "and it shapes everything from the animal's feeding schedule to the design of modern barns. "
) * 8

LIGHTHOUSE_TEXT = (
    "The lighthouse keeper had counted four hundred and twelve storms from the top of the "
    "tower, and every one of them had taught him something new about the sea. In the winter "
    "of his sixty-first year, a cargo ship ran aground on the northern shoal despite the "
    "light burning as brightly as ever, and the inquiry that followed brought a young "
    "engineer to the island with instruments the keeper did not recognize and opinions he "
    "did not share. "
) * 40


def hf_tokenizer_ids(model_dir: str, text: str, seq_len: int) -> torch.Tensor:
    """(1, seq_len) ids of text through the checkpoint's tokenizer.json (with its post-processor's special tokens)"""
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json")).encode(text).ids
    assert len(ids) >= seq_len, f"text too short: {len(ids)} < {seq_len} tokens"
    return torch.tensor([ids[:seq_len]], dtype = torch.long)


def truncated_checkpoint(src: str, dst: str, edit: Callable[[dict], None]) -> str:
    """Symlink farm of a checkpoint with an edited config.json (e.g. num_hidden_layers reduced, so a few layers of a
    model too large to run unquantized load on one device; both HF and exllamav3 ignore the surplus tensors)"""
    os.makedirs(dst, exist_ok = True)
    for name in os.listdir(src):
        if name != "config.json" and not os.path.exists(os.path.join(dst, name)):
            os.symlink(os.path.join(os.path.abspath(src), name), os.path.join(dst, name))
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    edit(cfg)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent = 2)
    return dst


def load_reference_module(path: str, name: str):
    """Import a reference implementation shipped with a checkpoint (a single .py file with no sibling imports) under
    a private module name, without touching sys.path"""
    import importlib.util
    import sys
    mod_name = f"_exl3_parity_ref_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module      # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def trim_layer_lists(cfg: dict, key: str | None = None):
    """Truncate per-layer list fields of a (sub-)config to num_hidden_layers (stub builders that only rewrote the layer
    count leave them full length, which transformers' config validation rejects)"""
    c = cfg[key] if key else cfg
    n = c["num_hidden_layers"]
    for k, v in c.items():
        if k.endswith("_types") and isinstance(v, list) and len(v) > n:
            c[k] = v[:n]


def token_ids(seq_len: int, vocab_size: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (1, seq_len), generator = g, dtype = torch.long)


def make_test_image(w: int = 613, h: int = 411):
    """Deterministic synthetic RGB image (gradients, shapes, a diagonal); the default odd size exercises resize
    schedules, canvas padding and edge padding of the vision preprocessors"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (int(255 * x / w), int(255 * y / h), int(255 * ((x + y) % 97) / 97))
    d = ImageDraw.Draw(img)
    d.ellipse((50, 40, 260, 210), fill = (240, 30, 30))
    d.rectangle((320, 180, 560, 360), fill = (30, 200, 60))
    d.line((0, 0, w, h), fill = (255, 255, 0), width = 9)
    return img


def mutate(objects, attr: str, value = None, where: Callable | None = None, transform: Callable | None = None) -> list:
    """Set `attr` to `value` (or transform(old value)) on every object that has it and satisfies where(obj). Returns
    [(object, old value)]; fails if nothing was mutated, so a renamed attribute can't make a mutation test vacuous"""
    saved = []
    for o in objects:
        if o is None or not hasattr(o, attr) or (where is not None and not where(o)):
            continue
        old = getattr(o, attr)
        saved.append((o, old))
        setattr(o, attr, transform(old) if transform is not None else value)
    assert saved, f"mutation target '{attr}' not found on any object"
    return saved


@contextlib.contextmanager
def mutated(objects, attr: str, value = None, where: Callable | None = None, transform: Callable | None = None):
    """mutate() for the duration of the context, restoring the original values on exit"""
    saved = mutate(objects, attr, value, where, transform)
    try:
        yield len(saved)
    finally:
        for o, old in saved:
            setattr(o, attr, old)
