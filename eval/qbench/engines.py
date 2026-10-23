"""
Inference backends. Each backend runs the full test set and hands per-row logits
[1, len, vocab] on the compute device to a callback, one row at a time, so full-model logits
never accumulate.

Effective bits-per-weight is computed to one convention across all formats, so the x-axis is an
apples-to-apples comparison:

  - layer bpw   = stored bits of the weight matrices (>= 2 dims) outside the embedding and the
                  output head, divided by their element count. Includes every storage overhead
                  of the format (scales, codebooks, sign flips, group metadata where they are
                  materialized as part of the tensor's storage).
  - excluded    = embeddings, norm weights, biases, and MoE router gates (tiny, always kept
                  high-precision, and represented inconsistently across formats).
  - head bpw    = the lm_head / output matrix; falls back to the embedding matrix's storage for
                  tied-embedding models.
  - vram_gb     = layer + head stored bytes.
"""

import os

import torch

from exllamav3.util.memory import free_mem
from exllamav3.util.progress import ProgressBar


class _Options(dict):
    """Model options with a record of which keys the backend actually read, so open_backend
    can echo the settings in force and flag options that no code path consumed"""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.used = set()
    def get(self, key, default = None):
        self.used.add(key)
        return super().get(key, default)
    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)
    def unused(self):
        return [k for k in self if k not in self.used]

# Substrings identifying MoE router gate matrices per naming scheme (2D but excluded: tiny and
# stored unquantized in some formats but quantized in others)
HF_ROUTER_KEYS = (".mlp.gate.weight", ".block_sparse_moe.gate.weight", ".router.weight", ".feed_forward.gate.weight")
GGUF_ROUTER_KEYS = ("ffn_gate_inp.weight",)


def apply_mult_noise(x: torch.Tensor, eps: float, gen: torch.Generator) -> torch.Tensor:
    n = torch.randn(x.shape, generator = gen, device = x.device, dtype = torch.float)
    return (x.float() * (1.0 + eps * n)).to(x.dtype)


class Exl3Backend:
    """Module-streamed exllamav3 pass (model_diff style): one module resident at a time"""

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        from exllamav3 import Config, Model
        self.device = device
        self.config = Config.from_directory(source)
        self.config.override_dynamic_seq_len(max_len)
        self.model = Model.from_config(self.config)
        self.info = None
        self.settings = {
            "architecture": getattr(self.config, "architecture", None),
            "max_len": max_len,
            "module_streamed": True,
        }

    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        from exllamav3.modules import Linear
        modules = self.model.modules
        # Per-row params, prepared by the architecture as Model.forward does before the module
        # chain (attention layout, recurrence, DeepSeek-V4's hash-routing ids / image chunks);
        # they carry state across modules, so each row keeps its own dict
        params_rows = [{} for _ in range(ids.shape[0])]
        states = [self.model.prepare_inputs(row, p) for row, p in zip(ids.split(1), params_rows)]
        gen = torch.Generator(device = self.device)
        gen.manual_seed(1)

        sum_bits = sum_numel = head_bits = head_numel = 0
        with ProgressBar("Streaming", len(modules)) as pb:
            for idx, module in enumerate(modules):
                self.config.stc.begin_deferred_load()
                module.load(self.device if not module.caps.get("prefer_cpu") else "cpu")
                self.config.stc.end_deferred_load()

                # Storage info while the module is resident. Biases are excluded from the
                # convention; the MoE routing gate is not a Linear in exllamav3 and is thus
                # excluded structurally
                if self.info is None:
                    stack = [module]
                    while stack:
                        m = stack.pop()
                        if isinstance(m, Linear):
                            bits = 8 * sum(
                                t.element_size() * t.numel()
                                for k, t in m.get_tensors().items()
                                if not k.endswith(".bias")
                            )
                            if m.key.endswith("lm_head"):
                                head_bits += bits
                                head_numel += m.weights_numel()
                            else:
                                sum_bits += bits
                                sum_numel += m.weights_numel()
                        stack.extend(getattr(m, "modules", []))

                logits_layer = idx == len(modules) - 1
                for r in range(len(states)):
                    params = params_rows[r]
                    params.setdefault("input_ids", ids[r:r + 1])
                    x = module.prepare_for_device(states[r], params)
                    x = module.forward(x, params)
                    # Modules may hand back a view of a shared g_tensor_cache workspace (mHC /
                    # GatedResidual streams); the next row's pass would overwrite it, so own it
                    if torch.is_tensor(x):
                        x = x.clone()
                    if noise_eps and idx < len(modules) - 2 and x.is_floating_point():
                        x = apply_mult_noise(x, noise_eps, gen)
                    if logits_layer:
                        callback(r, x)
                        states[r] = None
                    else:
                        states[r] = x
                    del x

                module.unload()
                self.config.stc.close()
                free_mem()
                pb.update(idx + 1)

        self.info = {
            "bpw_layer": sum_bits / max(sum_numel, 1),
            "bpw_head": head_bits / max(head_numel, 1),
            "vram_gb": (sum_bits + head_bits) / 8 / 1024 ** 3,
        }

    def close(self):
        self.model.unload()
        free_mem()


class TransformersBackend:
    """
    HF backend. Default: full load via accelerate device_map. With options: {streaming: true},
    the model is instead built as a weightless meta skeleton and each big submodule's weights
    are materialized from the safetensors shards just in time by forward hooks, then returned to
    meta - so peak VRAM is one decoder layer plus activations, and models far larger than the
    GPU (or system RAM) can serve as the reference. All rows are batched into a single forward,
    so each weight is read from disk exactly once per pass, and the head is applied per row so
    full [rows, len, vocab] logits never materialize. The model's own forward computes masks and
    rope, so no per-architecture layer-call knowledge is needed.

    Noise (for the self-noise floor) is injected with forward hooks on the decoder layer list,
    located as the largest ModuleList of same-type children; these compose with the streaming
    hooks unchanged.

    {time_step_limit: config} or {time_step_limit: [min, max]} overrides the runtime dt clamp of
    every Mamba2-style mixer (modules with a time_step_limit attribute). transformers' native
    NemotronH mixer takes its floor from time_step_min, the dt initialization range, where the
    checkpoint's time_step_limit says [0, inf] (as HF Mamba2 and exllamav3 read it); the floor
    makes the state drift along the sequence and every quant then appears to disagree with the
    reference past a few hundred tokens.
    """

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        from transformers import AutoModelForCausalLM
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32} \
            [options.get("dtype", "bfloat16")]
        self.device = device
        self.dtype = dtype
        self.streaming = options.get("streaming", False)
        # w4a4_sim: emulate the checkpoint's static activation quantization where it declares
        # input scales. NVFP4 expert GEMMs (ModelOpt W4A4 recipe): static per-expert input_scale
        # (calibrated global) + dynamic per-16-group e4m3 scales + e2m1 values, computed in fp32
        # and applied as fake-quant before the bf16 GEMMs. FP8 projections of a mixed export:
        # per-tensor static e4m3 fake-quant of the Linear input. Weight-side fidelity of the same
        # quantizer: 99.88% of stored e2m1 codes reproduced bit-exact from the bf16 originals
        self.w4a4_sim = options.get("w4a4_sim", False)
        if self.w4a4_sim and not self.streaming:
            raise ValueError("w4a4_sim is implemented on the streaming path (streaming: true)")
        self.shard_shims = []
        self.a4_wraps = []
        self.a4_hooks = []
        self.a4_count = [0, 0]     # expert modules, fp8 linears wrapped so far
        self.shard_handles = {}

        if self.streaming:
            import json
            import struct
            from accelerate import init_empty_weights
            from transformers import AutoConfig
            trc = options.get("trust_remote_code", True)
            config = AutoConfig.from_pretrained(source, trust_remote_code = trc)

            # Quantized checkpoints are dequantized on the fly by _get_tensor; the skeleton is
            # built unquantized (plain Linear modules), so the quantization config must not
            # reach from_config. Supported: compressed-tensors pack-quantized (AWQ-style int
            # groups), compressed-tensors mixed-precision / nvfp4-pack-quantized /
            # float-quantized (llm-compressor FP8-channel + NVFP4 mixes), and ModelOpt mixed
            # FP8 / NVFP4. Evaluation is weight-only in all cases: the deployed stacks'
            # activation quant (static or dynamic fp8/fp4) and fp8 kv cache are not emulated,
            # so results are the checkpoint's weight fidelity under bf16 compute
            self.ct_bits = None
            self.dequant_scaled = False
            self.scale_block = ()
            self.quant_desc = "none (checkpoint dtype)"
            qcfg = getattr(config, "quantization_config", None)
            if qcfg is not None:
                def q_get(key, default = None):
                    return qcfg.get(key, default) if isinstance(qcfg, dict) else getattr(qcfg, key, default)
                qm = q_get("quant_method")
                self.quant_desc = f"{qm} {q_get('format') or ''}".strip()
                if qm == "compressed-tensors":
                    fmt = q_get("format")
                    assert fmt in ("pack-quantized", "mixed-precision", "nvfp4-pack-quantized", "float-quantized"), \
                        f"Unsupported compressed-tensors format: {fmt}"
                    # ct_bits only drives the int pack-quantized decode; the actual scheme per
                    # tensor is decided by its sidecars (weight_global_scale -> nvfp4,
                    # weight_scale next to a plain weight -> fp8-channel)
                    self.ct_bits = 4
                    for g in (q_get("config_groups", {}) or {}).values():
                        w = (g.get("weights") if isinstance(g, dict) else getattr(g, "weights", None)) or {}
                        if isinstance(w, dict) and w.get("type", "int") == "int":
                            self.ct_bits = w.get("num_bits", 4)
                    self.dequant_scaled = True
                elif qm in ("modelopt", "modelopt_mixed"):
                    # ModelOpt HF export, uniform (NVFP4 or FP8) or MIXED_PRECISION (per-layer
                    # quant_algo: FP8 projections + NVFP4 experts, bf16 for the ignore list).
                    # The scheme of every tensor follows from its sidecars either way:
                    # weight_scale_2 -> NVFP4, fp8 weight + scalar weight_scale -> FP8
                    self.dequant_scaled = True
                    if qm == "modelopt_mixed":
                        import collections
                        algos = collections.Counter(
                            (v.get("quant_algo") if isinstance(v, dict) else getattr(v, "quant_algo", "?"))
                            for v in (q_get("quantized_layers", {}) or {}).values())
                        self.quant_desc = "modelopt mixed: " + ", ".join(f"{a} x{n}" for a, n in algos.most_common())
                elif qm == "fp8":
                    # AutoFP8/DeepSeek-style: fp8-e4m3 weights with blockwise (typically
                    # 128x128) weight_scale_inv, unquantized modules stay plain bf16
                    self.dequant_scaled = True
                    self.scale_block = tuple(q_get("weight_block_size", None) or ())
                    self.quant_desc = f"fp8 e4m3, scale blocks {'x'.join(map(str, self.scale_block)) or '128x128 (default)'}"
                elif qm == "mxfp4":
                    # gpt-oss-style: MoE expert tensors stored as <name>_blocks (packed fp4
                    # e2m1 pairs) + <name>_scales (e8m0); resolved per-tensor by sidecar
                    pass
                else:
                    raise ValueError(f"Streaming does not support this quantization_config: {qm}")
                delattr(config, "quantization_config")

            # include_buffers = False keeps non-persistent buffers (rope inv_freq etc., which
            # are not in the shards) materialized with their init values
            with init_empty_weights(include_buffers = False):
                try:
                    # Eager attention unless the project says otherwise: it is the reference arithmetic
                    # (fp32 softmax, explicit causal mask), and remote-code sdpa paths are not always
                    # causal (Ling 3.1's bundled modeling runs sdpa without a mask)
                    self.model = AutoModelForCausalLM.from_config(
                        config, dtype = dtype, trust_remote_code = trc,
                        attn_implementation = options.get("attn_implementation", "eager"))
                except ValueError:
                    # Multimodal wrappers not registered under CausalLM (Mistral3/Mistral4):
                    # the ConditionalGeneration class computes text-only logits fine, and the
                    # never-forwarded vision tower simply stays on meta
                    from transformers import AutoModelForImageTextToText
                    self.model = AutoModelForImageTextToText.from_config(
                        config, dtype = dtype, trust_remote_code = trc,
                        attn_implementation = options.get("attn_implementation", "eager"))
            self.model.eval()

            # The streamed head skips the CausalLM wrapper's forward, so post-head transforms
            # it would apply must be replicated (gemma-style logit softcapping, and MuseGlimmer's
            # output_multiplier pre-scale, applied BEFORE the softcap)
            text_config = getattr(config, "text_config", None) or config
            self.logit_softcap = (
                getattr(text_config, "final_logit_softcapping", None)
                or getattr(config, "final_logit_softcapping", None)
            )
            self.logit_multiplier = (
                getattr(text_config, "output_multiplier", None)
                or getattr(config, "output_multiplier", None)
                or 1.0
            )

            # Shard index: module-side tensor name -> (file, checkpoint tensor name). Checkpoint
            # layouts may differ from the module tree: transformers v5 declares per-model
            # conversions (renamings, and per-expert -> fused-experts merges) which
            # from_pretrained applies during load. Replicate that here: renamings fold into the
            # index; merge-type converters are kept and resolved on demand in _get_tensor
            import re
            from safetensors import safe_open
            self.source = source
            index_file = os.path.join(source, "model.safetensors.index.json")
            if os.path.exists(index_file):
                with open(index_file) as f:
                    raw_map = json.load(f)["weight_map"]
            else:
                single = os.path.join(source, "model.safetensors")
                with safe_open(single, framework = "pt", device = "cpu") as f:
                    raw_map = {k: "model.safetensors" for k in f.keys()}

            # Stored size per checkpoint tensor (from the shard headers), for exact bpw
            # accounting straight off the storage format
            self.tensor_nbytes = {}
            for fn in set(raw_map.values()):
                with open(os.path.join(source, fn), "rb") as f:
                    hdr_len = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hdr_len))
                for k, v in hdr.items():
                    if k != "__metadata__":
                        a, b = v["data_offsets"]
                        self.tensor_nbytes[k] = b - a

            # Conversion mappings may be registered under the composite model_type or, for
            # ConditionalGeneration wrappers whose CausalLM class is the inner text model, under
            # text_config.model_type (e.g. qwen3_5's model.language_model.* strip lives under
            # qwen3_5_text only)
            # Composite wrappers name the nested config freely (text_config, llm_config for
            # NemotronH Omni, ...), so take the model_type of every nested config object
            from transformers import PretrainedConfig
            model_types = [config.model_type]
            for sub in vars(config).values():
                if isinstance(sub, PretrainedConfig):
                    tmt = getattr(sub, "model_type", None)
                    if tmt and tmt not in model_types:
                        model_types.append(tmt)

            renames = []          # (source substring, target substring)
            self.converters = []  # (target pattern, [source patterns], [ops])
            try:
                from transformers.conversion_mapping import get_checkpoint_conversion_mapping
                for conv in [c for mt in model_types for c in (get_checkpoint_conversion_mapping(mt) or [])]:
                    sources = getattr(conv, "source_patterns", None)
                    targets = getattr(conv, "target_patterns", None)
                    ops = getattr(conv, "operations", None)
                    if not sources or not targets:
                        continue
                    if ops:
                        # Merge converters are matched below as literal suffixes with the
                        # engine's "experts.*.w1.weight" wildcard convention; transformers >= 5
                        # declares them as regex fragments (r"\.experts.*.w1.weight", target
                        # r"\.experts.gate_up_proj"), so unescape and drop the leading dot
                        def plain(pat):
                            return re.sub(r"\\(.)", r"\1", pat).lstrip(".")
                        self.converters.append((plain(targets[0]), [plain(sp) for sp in sources], ops))
                    else:
                        renames.append((sources[0], targets[0]))
            except ImportError:
                # Older transformers: fall back to the class-attr regex mapping
                for pat, repl in (getattr(type(self.model), "_checkpoint_conversion_mapping", None) or {}).items():
                    renames.append((pat, repl))

            # Renames are registered as aliases rather than replacements: some models' declared
            # renamings don't match their (vendored) module tree, so the original checkpoint
            # name stays resolvable either way
            self.tensor_index = {}
            for ck_name, fn in raw_map.items():
                self.tensor_index[ck_name] = (fn, ck_name)
                mod_name = ck_name
                for src, tgt in renames:
                    # Conversion sources are regex fragments; escaped literals ("mlp\.gate")
                    # must go through re.sub too, or they can never match anything (hy_v3's
                    # router/shared/bias renames are exactly this shape)
                    if src.startswith("^") or "(" in src or "\\" in src:
                        mod_name = re.sub(src, tgt, mod_name)
                    elif src in mod_name:
                        mod_name = mod_name.replace(src, tgt)
                if mod_name != ck_name:
                    self.tensor_index[mod_name] = (fn, ck_name)
                # Some conversion mappings (deepseek_v4) declare targets in the bare
                # base-model namespace and rely on the loader to add base_model_prefix;
                # alias the prefixed form so module-side lookups resolve. setdefault: real
                # top-level names (lm_head) must not be shadowed
                base_prefix = getattr(self.model, "base_model_prefix", "")
                if base_prefix:
                    self.tensor_index.setdefault(f"{base_prefix}.{mod_name}", (fn, ck_name))

            # Generic fallback for composite checkpoints whose model doesn't declare the
            # container strip in its conversion mapping: alias the text stack into the CausalLM
            # namespace (declared renames take precedence via setdefault)
            for ck_name, fn in raw_map.items():
                if ck_name.startswith("model.language_model."):
                    alias = "model." + ck_name[len("model.language_model."):]
                    self.tensor_index.setdefault(alias, (fn, ck_name))

            # Wrappers nesting a bare text model under language_model keep the text model's own
            # base prefix in the checkpoint ("language_model.model.*", old-style Mistral3/4);
            # from_pretrained strips it when nesting, so alias the stripped form of every
            # resolved name too
            for mod_name in list(self.tensor_index.keys()):
                if ".language_model.model." in mod_name:
                    self.tensor_index.setdefault(
                        mod_name.replace(".language_model.model.", ".language_model.", 1),
                        self.tensor_index[mod_name])

            # Real (init-valued) buffers up front on the compute device
            for m in self.model.modules():
                for k, b in m._buffers.items():
                    if b is not None and not b.is_meta:
                        m._buffers[k] = b.to(device)

            # Persistent buffers stored in the checkpoint (e.g. gemma4's per-layer scalars) are
            # not covered by parameter streaming; they are small, so load them up front. Missing
            # them is silent and catastrophic - the module keeps its init value
            for bn, _ in self.model.named_buffers():
                if bn in self.tensor_index:
                    t = self._read_shard(bn)
                    if t.is_floating_point() and t.dtype != torch.float32:
                        t = t.to(device, dtype)
                    else:
                        t = t.to(device)
                    owner_name, _, leaf = bn.rpartition(".")
                    owner = self.model.get_submodule(owner_name) if owner_name else self.model
                    owner._buffers[leaf] = t

            self.prefix = {id(m): n for n, m in self.model.named_modules()}
            # Tied-head fallback: the output embedding's weight is not in the shards when tied
            _, head = self._text_parts()
            embed = self.model.get_input_embeddings()
            self.head_weight_name = f"{self.prefix[id(head)]}.weight" if head is not None else None
            self.embed_weight_name = f"{self.prefix[id(embed)]}.weight"
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    dtype = dtype,
                    device_map = options.get("device_map", "auto"),
                    trust_remote_code = options.get("trust_remote_code", True),
                )
            except ValueError:
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    source,
                    dtype = dtype,
                    device_map = options.get("device_map", "auto"),
                    trust_remote_code = options.get("trust_remote_code", True),
                )
            self.model.eval()

        self.time_step_limit = self._apply_time_step_limit(options.get("time_step_limit"))

        self.settings = {
            "dtype": str(dtype).replace("torch.", ""),
            "streaming": self.streaming,
            "trust_remote_code": options.get("trust_remote_code", True),
            # Requested vs what transformers actually installed on the model config
            "attn_implementation": f"{options.get('attn_implementation', 'eager' if self.streaming else 'default')}"
                                   f" (model: {getattr(self.model.config, '_attn_implementation', '?')})",
        }
        if self.streaming:
            self.settings["dequant"] = self.quant_desc
        else:
            self.settings["device_map"] = options.get("device_map", "auto")
        if self.w4a4_sim:
            n_scales = sum(k.endswith(".input_scale") for k in self.tensor_index)
            if not n_scales:
                raise ValueError("w4a4_sim: the checkpoint carries no input_scale sidecars")
            self.settings["w4a4_sim"] = f"{n_scales} input scales in checkpoint"
        if self.time_step_limit is not None:
            self.settings["time_step_limit"] = self.time_step_limit
        self.settings["modeling"] = self._modeling_source(source)
        import transformers
        self.settings["transformers"] = transformers.__version__

        # Biases and norms are 1D (excluded by the ndim test); router gates excluded by name.
        # In streaming mode, bits come from the checkpoint's actual stored bytes (including
        # quantization metadata like scales), so quantized formats report their true bitrate.
        # Packed sub-byte formats (compressed-tensors nvfp4/int-pack, GPTQ/AWQ qweight) store
        # several weights per element: translate the packed numel to the true weight count, and
        # count scale/zero-point tensors as bits belonging to the owning weight, not as weights
        qcfg = getattr(self.model.config, "quantization_config", None)
        if qcfg is not None and not isinstance(qcfg, dict):
            qcfg = getattr(qcfg, "to_dict", lambda: {})()
        qcfg = qcfg or {}
        pack_num_bits = qcfg.get("bits")
        for g in (qcfg.get("config_groups") or {}).values():
            nb = isinstance(g, dict) and (g.get("weights") or {}).get("num_bits")
            if nb: pack_num_bits = nb
        qmeta_suffixes = (
            "_scale", "_scale_2", "_scale_inv", "_global_scale", "_zero_point", "_shape",
            "_g_idx", ".qzeros", ".scales", ".g_idx",
        )

        sum_bits = sum_numel = head_bits = head_numel = 0
        embed_bits = embed_numel = 0
        seen = set()
        named = list(self.model.named_parameters())
        if qcfg:
            # Quantized loaders may register packed tensors as buffers
            named += [nb for nb in self.model.named_buffers() if nb[0] not in {n for n, _ in named}]
        for name, p in named:
            if name in seen:
                continue
            seen.add(name)
            if name.endswith(".bias") or name.endswith("_bias"):
                continue
            # Multimodal encoder stacks don't contribute to text logits; the exllamav3 engine
            # walks only the text component, so exclude them here for a comparable bpw_layer
            if any(k in name for k in ("vision", "visual", "mm_projector", "multi_modal", "audio_tower")):
                continue
            # Match the exl3/GGUF conventions (Linear-module weights only): hyper-connection
            # matrices, short-conv kernels and scalar shared-expert gates are excluded there
            if "hyper_connection" in name or "shared_expert_gate" in name \
                    or name.endswith("conv1d.weight"):
                continue
            is_qmeta = any(name.endswith(s) for s in qmeta_suffixes)
            if (name.endswith("_packed") or name.endswith(".qweight")) and pack_num_bits:
                numel = p.numel() * (p.element_size() * 8 // pack_num_bits)
            elif is_qmeta:
                numel = 0
            else:
                numel = p.numel()
            bits = None
            if self.streaming:
                bits = self._stored_bits(name)
            if bits is None:
                bits = p.numel() * p.element_size() * 8
            if "embed" in name:
                embed_bits += bits
                embed_numel += numel
                continue
            if "lm_head" in name or "output" == name.split(".")[0]:
                head_bits += bits
                head_numel += numel
            elif (p.ndim >= 2 or is_qmeta) and not any(k in name for k in HF_ROUTER_KEYS):
                sum_bits += bits
                sum_numel += numel
        if head_numel == 0:
            # Tied embeddings: the head is served by the embedding matrix
            head_bits, head_numel = embed_bits, embed_numel
        self.info = {
            "bpw_layer": sum_bits / max(sum_numel, 1),
            "bpw_head": head_bits / max(head_numel, 1),
            "vram_gb": (sum_bits + head_bits) / 8 / 1024 ** 3,
        }

    def _apply_time_step_limit(self, spec):
        """Set the runtime dt clamp of every mixer that has one. spec: "config" takes the first
        time_step_limit found on the model's config or its nested configs; a [min, max] pair is
        used as given (max may be "inf"). Returns the limit applied, as a string for the settings
        printout, or None"""
        if spec is None:
            return None
        from transformers import PretrainedConfig
        if spec == "config":
            cfgs = [self.model.config] + [v for v in vars(self.model.config).values() if isinstance(v, PretrainedConfig)]
            lim = next((getattr(c, "time_step_limit") for c in cfgs if getattr(c, "time_step_limit", None) is not None), None)
            if lim is None:
                raise ValueError("time_step_limit: config declares no time_step_limit")
        else:
            lim = spec
        lim = tuple(float(v) for v in lim)
        assert len(lim) == 2, "time_step_limit: [min, max]"
        n = 0
        for m in self.model.modules():
            if hasattr(m, "time_step_limit"):
                m.time_step_limit = lim
                n += 1
        if not n:
            raise ValueError("time_step_limit: no module with a time_step_limit attribute")
        return f"[{lim[0]:g}, {lim[1]:g}] ({n} mixers)"

    def _text_parts(self):
        """(base, head): the module whose forward returns last_hidden_state, and the output
        embedding. Multimodal wrappers that nest a bare text model expose both through the usual
        base_model / get_output_embeddings; wrappers that nest a complete causal LM under
        language_model (NemotronH Omni: a *ForCausalLM with its own lm_head, and no
        base_model_prefix of their own) are descended into"""
        m = self.model
        while True:
            base = getattr(m, "base_model", m)
            if base is not m:
                m = base
                continue
            lm = getattr(m, "language_model", None)
            if lm is None or getattr(lm, "get_output_embeddings", lambda: None)() is None:
                break
            m = lm
        head = self.model.get_output_embeddings()
        if head is None:
            inner = self.model
            while head is None and getattr(inner, "language_model", None) is not None:
                inner = inner.language_model
                head = inner.get_output_embeddings()
        return m, head

    def _decoder_layers(self):
        """The decoder layer list: the largest ModuleList under the text base that no other
        candidate ModuleList contains. Per-expert MoE implementations (Ling) hold a ModuleList
        of hundreds of experts inside every layer, which would win on size alone; the layer
        list is the outermost one. Mixed child types are allowed (Ling appends its MTP layer
        to the list). Searching the text base only keeps a vision tower's block list (deeper
        than a short text stack) out of the running"""
        base, _ = self._text_parts()
        lists = [m for m in base.modules() if isinstance(m, torch.nn.ModuleList) and len(m) >= 4]
        inner = set()
        for m in lists:
            for sub in m.modules():
                if sub is not m and isinstance(sub, torch.nn.ModuleList):
                    inner.add(id(sub))
        outer = [m for m in lists if id(m) not in inner]
        if not outer:
            raise RuntimeError("Could not locate decoder layer list for noise injection")
        return max(outer, key = len)

    def _noise_hooks(self, noise_eps):
        gens = {}
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.device not in gens:
                g = torch.Generator(device = out.device)
                g.manual_seed(1)
                gens[out.device] = g
            out = apply_mult_noise(out, noise_eps, gens[out.device])
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return [layer.register_forward_hook(hook) for layer in self._decoder_layers()]

    # ------ streaming machinery

    CT_SUFFIXES = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx",
                   "weight_global_scale", "input_global_scale")

    def _read_shard(self, name):
        fn, ck_name = self.tensor_index[name]
        fn = os.path.join(self.source, fn)
        if fn not in self.shard_handles:
            from safetensors import safe_open
            self.shard_handles[fn] = safe_open(fn, framework = "pt", device = "cpu")
        return self.shard_handles[fn].get_tensor(ck_name)

    def _nbytes(self, name):
        return self.tensor_nbytes[self.tensor_index[name][1]]

    def _dequant_ct(self, stem):
        """Dequantize a compressed-tensors pack-quantized weight: sequentially nibble-packed
        signed ints in int32 words, per-group scales, optional zero point"""
        bits = self.ct_bits or 4
        packed = self._read_shard(f"{stem}.weight_packed")          # int32 [out, in * bits/32]
        scale = self._read_shard(f"{stem}.weight_scale").float()    # [out, groups]
        shape = self._read_shard(f"{stem}.weight_shape").tolist()   # [out, in]
        shifts = torch.arange(0, 32, bits, dtype = torch.int32)
        mask = (1 << bits) - 1
        q = (packed.unsqueeze(-1) >> shifts) & mask
        q = q.flatten(1)[:, :shape[1]]
        q = q - (1 << (bits - 1))  # values are stored offset-binary (e.g. int4 as value + 8)
        zp_name = f"{stem}.weight_zero_point"
        group = shape[1] // scale.shape[1]
        if zp_name in self.tensor_index:
            zp = self._read_shard(zp_name)
            if zp.dtype == torch.int32:
                # Packed offset-binary like the weights, but the packing axis differs by
                # tensor: zero points are [out, groups] packed along whichever dim makes the
                # int32 count fit (observed: along out, unlike the weights)
                per_word = 32 // bits
                z = (zp.unsqueeze(-1) >> shifts) & mask
                if zp.shape[0] * per_word == shape[0]:      # packed along dim 0 (out)
                    zp = z.permute(0, 2, 1).reshape(-1, zp.shape[1])[:shape[0]]
                else:                                       # packed along dim 1 (groups)
                    zp = z.flatten(1)[:, :scale.shape[1]]
                zp = zp - (1 << (bits - 1))
            q = q - zp.repeat_interleave(group, dim = 1)
        return q.to(torch.float32) * scale.repeat_interleave(group, dim = 1)

    # fp4 e2m1 magnitudes for codes 0-7; codes 8-15 mirror negative (sign is the msb)
    NVFP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

    def _modeling_source(self, source):
        """Where the model class came from: the transformers module, or for remote code the
        cached copy, checked against the same-named file in the model directory so a patched
        modeling file can be confirmed to be the one in use"""
        import inspect, hashlib
        cls = type(self.model)
        try:
            path = inspect.getsourcefile(cls)
        except TypeError:
            path = None
        if not path:
            return cls.__module__
        if cls.__module__.startswith("transformers."):
            return f"{cls.__module__}.{cls.__name__}"
        local = os.path.join(source, os.path.basename(path))
        if os.path.isfile(local):
            same = hashlib.sha256(open(path, "rb").read()).digest() == hashlib.sha256(open(local, "rb").read()).digest()
            return f"{local} ({'in use' if same else 'DIFFERS from the loaded copy ' + path})"
        return path

    def _scale_float(self, s):
        """Scale sidecar to fp32. E8M0 exponent bytes (DeepSeek-V4 ue8m0, stored as uint8 or
        torch.float8_e8m0fnu) decode as 2^(byte - 127); everything else casts directly."""
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if s.dtype == torch.uint8 or (e8m0 is not None and s.dtype == e8m0):
            return (s.view(torch.uint8).float() - 127.0).exp2()
        return s.float()

    def _dequant_scaled(self, name):
        """Dequantize a weight stored under its plain name with a weight_scale sidecar:
        ModelOpt NVFP4 (uint8-packed e2m1 nibble pairs low-first, fp8-e4m3 scale per group,
        fp32 global scale) when weight_scale_2 exists, else FP8 (e4m3 values; per-tensor
        scalar for ModelOpt, [out, 1] channel scale for compressed-tensors float-quantized).
        FP8 with blockwise weight_scale_inv (DeepSeek-style 128x128 blocks; despite the name,
        dequantization MULTIPLIES by it) is handled first, as is the equivalent DeepSeek-V4
        "<stem>.scale" sidecar convention (fp8 128x128 grid on dense weights, [out, in/32]
        e8m0 grid on int8-packed fp4 expert weights; block shape derives from the grid).
        All conventions validated against the bf16 originals."""
        w = self._read_shard(name)
        if w.dtype == torch.int8:
            # DeepSeek-V4 fp4 experts: two e2m1 values per byte, low nibble first
            u8 = w.view(torch.uint8)
            lut = self.NVFP4_LUT.to(w.device)
            w = torch.stack([lut[(u8 & 0x0F).long()], lut[(u8 >> 4).long()]], dim = -1).flatten(1)
        si = None
        if f"{name}_scale_inv" in self.tensor_index:
            si = self._scale_float(self._read_shard(f"{name}_scale_inv"))
        elif name.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.tensor_index:
            si = self._scale_float(self._read_shard(f"{name[:-len('.weight')]}.scale"))
        if si is not None:
            # Block grid of any rank: per-tensor scalar (Mistral-Small-4 dense weights),
            # per-expert [E, 1, 1] on fused 3D expert stacks, or the 2D 128x128 DeepSeek grid
            if si.ndim > 0:
                assert si.ndim == w.ndim, f"{name}: scale rank {si.ndim} vs weight rank {w.ndim}"
                for d in range(w.ndim):
                    if si.shape[d] != w.shape[d]:
                        if w.shape[d] % si.shape[d] == 0:
                            bs = w.shape[d] // si.shape[d]
                        else:
                            # The dimension ends in a partial block (Ling's 576-row
                            # kv_a_proj_with_mqa under a 5-row grid), so the ratio is not
                            # the block size: take it from the checkpoint's block shape
                            # (last dims), else the DeepSeek 128 default, and check coverage
                            blk = self.scale_block[d - (w.ndim - len(self.scale_block))] \
                                if 0 <= d - (w.ndim - len(self.scale_block)) < len(self.scale_block) else 128
                            bs = int(blk)
                            assert (w.shape[d] + bs - 1) // bs == si.shape[d], \
                                f"{name}: {si.shape[d]} scale blocks of {bs} do not cover dim {d} of {tuple(w.shape)}"
                        si = si.repeat_interleave(bs, dim = d).narrow(d, 0, w.shape[d])
            return w.float() * si
        scale = self._read_shard(f"{name}_scale").float()
        if scale.ndim == 1:
            scale = scale.unsqueeze(1)
        if f"{name}_scale_2" in self.tensor_index:
            ws2 = self._read_shard(f"{name}_scale_2").float()
            q = torch.stack([(w & 0xF).long(), (w >> 4).long()], dim = -1).flatten(1)
            group = q.shape[1] // scale.shape[1]
            return self.NVFP4_LUT[q] * scale.repeat_interleave(group, dim = 1) * ws2
        return w.float() * scale

    def _dequant_ct_nvfp4(self, stem):
        """Dequantize a compressed-tensors nvfp4-pack-quantized weight: uint8-packed e2m1
        nibble pairs low-first, fp8-e4m3 scale per group, and a global scale the stored
        per-group scales are DIVIDED by (it maps them into fp8 range at quantization time)"""
        packed = self._read_shard(f"{stem}.weight_packed")
        scale = self._read_shard(f"{stem}.weight_scale").float()
        gs = self._read_shard(f"{stem}.weight_global_scale").float()
        q = torch.stack([(packed & 0xF).long(), (packed >> 4).long()], dim = -1).flatten(1)
        group = q.shape[1] // scale.shape[1]
        return self.NVFP4_LUT[q] * scale.repeat_interleave(group, dim = 1) / gs

    def _stored_bits(self, name):
        """Actual stored size in the checkpoint for a module-side tensor name, including any
        quantization metadata; None if the name cannot be resolved"""
        import re
        if name in self.tensor_index:
            total = 8 * self._nbytes(name)
            if name.endswith(".weight"):
                # ModelOpt-style scale sidecars stored next to the weight itself
                stem = name[:-len(".weight")]
                for s in ("weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale", "input_global_scale", "scale"):
                    if f"{stem}.{s}" in self.tensor_index:
                        total += 8 * self._nbytes(f"{stem}.{s}")
            # Underscore sidecars on suffix-less fused tensors (fp8 expert stacks)
            for s in ("_scale", "_scale_inv"):
                if f"{name}{s}" in self.tensor_index:
                    total += 8 * self._nbytes(f"{name}{s}")
            return total
        if name.endswith(".weight"):
            stem = name[:-len(".weight")]
            bits = [8 * self._nbytes(f"{stem}.{s}") for s in self.CT_SUFFIXES if f"{stem}.{s}" in self.tensor_index]
            if bits:
                return sum(bits)
        if f"{name}_blocks" in self.tensor_index:
            return 8 * (self._nbytes(f"{name}_blocks") + self._nbytes(f"{name}_scales"))
        if name == self.head_weight_name and self.embed_weight_name in self.tensor_index:
            return 8 * self._nbytes(self.embed_weight_name)
        for target, sources, ops in self.converters:
            if not (name == target or name.endswith("." + target)):
                continue
            prefix = name[:-len(target)]
            total = 0
            for sp in sources:
                pat = re.compile(
                    "^" + re.escape(prefix + sp).replace(r"\*", r"\d+") + r"(_packed|_scale|_zero_point|_shape|_g_idx)?$"
                )
                total += sum(8 * self._nbytes(k) for k in self.tensor_index if pat.match(k))
            if total:
                return total
        fm = re.match(r"^(.*\.experts)\.(gate_up_proj|up_proj|down_proj)$", name)
        if fm:
            prefix, kind = fm.group(1), fm.group(2)
            parts = {"down_proj": ("down",), "up_proj": ("up",), "gate_up_proj": ("gate", "up")}[kind]
            pat = re.compile(
                "^" + re.escape(prefix) + r"\.\d+\.(" + "|".join(parts) +
                r")_proj\.weight(_packed|_scale|_scale_inv|_zero_point|_shape|_g_idx)?$")
            total = sum(8 * self._nbytes(k) for k in self.tensor_index if pat.match(k))
            if total:
                return total
        return None

    def _get_tensor(self, name):
        import re
        if name in self.tensor_index:
            # ModelOpt, ct-float-quantized and fp8 keep the quantized tensor under the plain
            # .weight name; the scale sidecar marks it as needing dequant. Fused fp8 expert
            # tensors (Mistral-Small-4 experts.gate_up_proj etc.) carry the same _scale_inv
            # sidecar without a .weight suffix
            if self.dequant_scaled and (
                    f"{name}_scale" in self.tensor_index or f"{name}_scale_inv" in self.tensor_index
                    or (name.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.tensor_index)):
                return self._dequant_scaled(name)
            return self._read_shard(name)
        if name == self.head_weight_name and self.embed_weight_name in self.tensor_index:
            return self._read_shard(self.embed_weight_name)  # tied embeddings

        # gpt-oss-style MXFP4 (<name>_blocks + <name>_scales, e.g. mlp.experts.gate_up_proj):
        # transformers' own converter handles the fp4-pair unpack, e8m0 ldexp and the transpose
        # into module orientation
        if f"{name}_blocks" in self.tensor_index:
            from transformers.integrations.mxfp4 import convert_moe_packed_tensors
            return convert_moe_packed_tensors(
                self._read_shard(f"{name}_blocks"), self._read_shard(f"{name}_scales"))

        # compressed-tensors packed weight: nvfp4 when a global scale exists, int otherwise
        if name.endswith(".weight"):
            stem = name[:-len(".weight")]
            if f"{stem}.weight_packed" in self.tensor_index:
                if f"{stem}.weight_global_scale" in self.tensor_index:
                    return self._dequant_ct_nvfp4(stem)
                return self._dequant_ct(stem)

        # Per-expert checkpoints vs fused-expert module trees (mistral4 AWQ/compressed-tensors:
        # experts.{i}.{gate,up,down}_proj vs experts.gate_up_proj) with no declared conversion
        # mapping: stack the per-expert tensors, resolving each recursively so quantized
        # formats dequantize on the way in. Module orientation is (E, out, in), gate rows first
        fm = re.match(r"^(.*\.experts)\.(gate_up_proj|down_proj)$", name)
        if fm:
            prefix, kind = fm.group(1), fm.group(2)
            probe = "down" if kind == "down_proj" else "gate"
            e = 0
            while (f"{prefix}.{e}.{probe}_proj.weight" in self.tensor_index or
                   f"{prefix}.{e}.{probe}_proj.weight_packed" in self.tensor_index):
                e += 1
            if e:
                if kind == "down_proj":
                    return torch.stack([self._get_tensor(f"{prefix}.{i}.down_proj.weight") for i in range(e)])
                return torch.stack([
                    torch.cat([self._get_tensor(f"{prefix}.{i}.gate_proj.weight"),
                               self._get_tensor(f"{prefix}.{i}.up_proj.weight")], dim = 0)
                    for i in range(e)])

        # Merge-type conversion (fused MoE experts): gather the per-expert source tensors for
        # this prefix, stack each source group (MergeModulelist), then concatenate groups
        # (Concatenate) if the converter has more than one. Sources resolve recursively, so
        # quantized per-expert weights dequantize on the way in
        for target, sources, ops in self.converters:
            if not (name == target or name.endswith("." + target)):
                continue
            prefix = name[:-len(target)]
            merge_dim = next((getattr(o, "dim", 0) for o in ops if type(o).__name__ == "MergeModulelist"), 0)
            cat_dim = next((getattr(o, "dim", 1) for o in ops if type(o).__name__ == "Concatenate"), None)
            groups = []
            for sp in sources:
                base = prefix + sp
                if "*" not in base:
                    # Literal (non-wildcard) source, e.g. glm5_next's per-head conv weights
                    # q/k/v_conv1d.weight -> Concatenate into conv1d.weight: resolve the
                    # single tensor directly (recursively, so quantized formats dequantize)
                    groups.append(self._get_tensor(base))
                    continue
                if base.endswith(".weight"):
                    stem_re = re.compile(
                        "^" + re.escape(base[:-len(".weight")]).replace(r"\*", r"(\d+)") + r"\.weight(_packed)?$"
                    )
                    matched = {}
                    for key in self.tensor_index:
                        m = stem_re.match(key)
                        if m:
                            matched[int(m.group(1))] = key[:key.rindex(".weight")]
                    if not matched:
                        raise KeyError(f"No checkpoint tensors match {base}")
                    tensors = [self._get_tensor(matched[i] + ".weight") for i in sorted(matched)]
                else:
                    pat = re.compile("^" + re.escape(base).replace(r"\*", r"(\d+)") + "$")
                    matched = sorted(
                        (int(m.group(1)), key)
                        for key in self.tensor_index
                        if (m := pat.match(key))
                    )
                    if not matched:
                        raise KeyError(f"No checkpoint tensors match {base}")
                    tensors = [self._read_shard(k) for _, k in matched]
                groups.append(torch.stack(tensors, dim = merge_dim))
            return torch.cat(groups, dim = cat_dim) if cat_dim is not None and len(groups) > 1 else groups[0]

        raise KeyError(f"{name} not found in checkpoint shards")

    def _plain_fp32(self, name):
        """True for tensors stored fp32 and read verbatim (no dequant): these are exactly the
        ones from_pretrained's _keep_in_fp32_modules lists keep in fp32 (hc mixers, sinks,
        position biases, norms), so the streamed skeleton must too or mHC/routing amplify the
        bf16 rounding into a visible reference deviation."""
        if name not in self.tensor_index:
            return False
        if f"{name}_scale" in self.tensor_index or f"{name}_scale_inv" in self.tensor_index:
            return False
        if name.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.tensor_index:
            return False
        return True

    def _materialize(self, module):
        prefix = self.prefix[id(module)]
        sd = {}
        shims = []
        for pn, _ in module.named_parameters():
            full = f"{prefix}.{pn}" if prefix else pn
            if full not in self.tensor_index and full.endswith(".weight") \
                    and f"{full[: -len('.weight')]}.shard_0.weight" in self.tensor_index:
                # Parameter stored as shard_N pieces too large to concatenate (Qwen4Exp's
                # ~100 GB n-gram table): serve it from the mmapped shards instead
                shims.append(pn)
                continue
            t = self._get_tensor(full)
            if t.is_floating_point() and not (t.dtype == torch.float32 and self._plain_fp32(full)):
                t = t.to(self.device, self.dtype)
            else:
                t = t.to(self.device)
            sd[pn] = t
        module.load_state_dict(sd, strict = False, assign = True)
        for pn in shims:
            self._install_shard_shim(module, prefix, pn)
        for pn, p in module.named_parameters():
            if p.is_meta:
                raise RuntimeError(f"No checkpoint tensor for {self.prefix[id(module)]}.{pn}")
        if self.w4a4_sim:
            self._install_a4_wraps(module, prefix)

    NVFP4_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def _nvfp4_fq(self, x, gs):
        """NVFP4 fake-quant: per-16-group e4m3 scale = e4m3(amax / 6 / gs), values rounded to
        the e2m1 grid. Returns x.dtype."""
        lut = self.NVFP4_POS.to(x.device)
        mid = (lut[1:] + lut[:-1]) / 2
        xr = x.float().reshape(-1, 16)
        amax = xr.abs().amax(dim = 1, keepdim = True)
        sg = (amax / 6.0 / gs).to(torch.float8_e4m3fn).float() * gs
        sg = torch.where(sg == 0, torch.ones_like(sg), sg)
        q = xr / sg
        v = lut[torch.bucketize(q.abs().clamp(max = 6.0), mid)] * q.sign()
        return (v * sg).reshape(x.shape).to(x.dtype)

    def _fp8_fq(self, x, s):
        """Static per-tensor FP8 fake-quant (ModelOpt FP8 recipe): e4m3 values of x / s, saturated
        at the e4m3 range, scaled back. Returns x.dtype."""
        return ((x.float() / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s).to(x.dtype)

    def _install_a4_wraps(self, module, prefix):
        """Emulate the checkpoint's static activation quantization where it declares input
        scales. Fused-expert modules with a forward(hidden, top_k_index, top_k_weights), gated
        (Qwen3Next-style gate_up_proj) or gateless (NemotronH up_proj), get per-expert NVFP4
        fake-quant on both GEMM inputs when the checkpoint carries per-expert input_scale
        tensors; plain fp8 Linears (scalar weight_scale, no weight_scale_2) get a per-tensor e4m3
        fake-quant of their input. Modules without input scales are left alone and counted
        nowhere, so a4_count stays 0 for a checkpoint the simulation does not cover."""
        import torch.nn.functional as F
        for name, sub in module.named_modules():
            pset = dict(sub.named_parameters(recurse = False))
            # The root module (a materialized lm_head) comes back with an empty name
            stem = ".".join(part for part in (prefix, name) if part)
            gated = "gate_up_proj" in pset
            if (gated or "up_proj" in pset) and "down_proj" in pset and pset["down_proj"].dim() == 3:
                E = pset["down_proj"].shape[0]
                first = "gate" if gated else "up"
                if f"{stem}.0.{first}_proj.input_scale" not in self.tensor_index:
                    continue
                us = torch.tensor([float(self._read_shard(f"{stem}.{e}.{first}_proj.input_scale"))
                                   for e in range(E)])
                ds = torch.tensor([float(self._read_shard(f"{stem}.{e}.down_proj.input_scale"))
                                   for e in range(E)])
                orig = sub.forward

                def a4_forward(hidden_states, top_k_index, top_k_weights, _m = sub, _us = us, _ds = ds, _gated = gated):
                    final = torch.zeros_like(hidden_states)
                    with torch.no_grad():
                        # One spare class: routers that pad with the expert count as a sentinel
                        mask = F.one_hot(top_k_index, num_classes = _m.num_experts + 1).permute(2, 1, 0)
                        hit = torch.greater(mask.sum(dim = (-1, -2)), 0).nonzero()
                    for e in hit:
                        e = int(e[0])
                        if e == _m.num_experts:
                            continue
                        pos, tok = torch.where(mask[e])
                        cs = self._nvfp4_fq(hidden_states[tok], float(_us[e]))
                        if _gated:
                            gate, up = F.linear(cs, _m.gate_up_proj[e]).chunk(2, dim = -1)
                            h = _m.act_fn(gate) * up
                        else:
                            h = _m.act_fn(F.linear(cs, _m.up_proj[e]))
                        h = self._nvfp4_fq(h, float(_ds[e]))
                        h = F.linear(h, _m.down_proj[e])
                        h = h * top_k_weights[tok, pos, None]
                        final.index_add_(0, tok, h.to(final.dtype))
                    return final

                sub.forward = a4_forward
                self.a4_wraps.append((sub, orig))
                self.a4_count[0] += 1
            elif isinstance(sub, torch.nn.Linear) and f"{stem}.input_scale" in self.tensor_index \
                    and f"{stem}.weight_scale" in self.tensor_index and f"{stem}.weight_scale_2" not in self.tensor_index:
                s8 = float(self._read_shard(f"{stem}.input_scale"))
                self.a4_hooks.append(sub.register_forward_pre_hook(
                    lambda m, args, _s = s8: (self._fp8_fq(args[0], _s),) + tuple(args[1:])))
                self.a4_count[1] += 1

    def _install_shard_shim(self, module, prefix, pn):
        stem_local = pn[: -len(".weight")]
        stem_full = f"{prefix}.{stem_local}" if prefix else stem_local
        shards = []
        while f"{stem_full}.shard_{len(shards)}.weight" in self.tensor_index:
            shards.append(self._read_shard(f"{stem_full}.shard_{len(shards)}.weight"))
        owner = module.get_submodule(stem_local)
        parent_path, _, owner_name = stem_local.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        # fp8-stored tables (Qwen3.8-Flash-Next nvfp4 export) carry one global weight_scale
        scale = None
        if f"{stem_full}.weight_scale" in self.tensor_index:
            scale = self._read_shard(f"{stem_full}.weight_scale").float().flatten()[0].item()
        parent._modules[owner_name] = _ShardGatherEmbedding(shards, self.dtype, scale)
        self.shard_shims.append((parent, owner_name, owner))

    def _dematerialize(self, module):
        while self.a4_wraps:
            sub, orig = self.a4_wraps.pop()
            sub.forward = orig
        while self.a4_hooks:
            self.a4_hooks.pop().remove()
        while self.shard_shims:
            parent, name, orig = self.shard_shims.pop()
            parent._modules[name] = orig            # original keeps its meta weight
        for sub in module.modules():
            for pn, p in list(sub._parameters.items()):
                if p is not None and not p.is_meta:
                    sub._parameters[pn] = torch.nn.Parameter(p.to("meta"), requires_grad = False)

    @torch.inference_mode()
    def _run_streaming(self, ids: torch.Tensor, callback, noise_eps: float = None):
        base, head = self._text_parts()
        layers = self._decoder_layers()
        embed = self.model.get_input_embeddings()

        # Hook the embedding, every decoder layer, and every other weight-carrying module found
        # by walking the base tree - recursing into containers that hold the special modules
        # (multimodal wrappers nest the text stack one level down) instead of hooking them
        # wholesale, so per-layer streaming is preserved. The head is streamed manually below
        special = {id(embed)} | {id(m) for m in layers}
        def contains_special(m):
            return any(id(x) in special for x in m.modules())
        extra = []
        def walk(m):
            for child in m.children():
                if id(child) in special:
                    continue
                if not any(True for _ in child.parameters()):
                    continue
                if contains_special(child):
                    walk(child)
                else:
                    extra.append(child)
        walk(base)
        hook_modules = [embed] + list(layers) + extra

        # Layer-major streaming: every weight is read exactly ONCE for the whole trace, and
        # every row runs unpadded at its own length. Each row's forward runs in its own
        # thread; the module hooks act as a scheduler that inverts the loop order without
        # knowing any model's dataflow: a hooked module materializes once, the rows pass
        # through it one at a time (other threads suspend at the pre-hook), then it
        # dematerializes and the next module (discovered dynamically from whichever module
        # the lead row requests next) begins. Suspended rows hold only their inter-module
        # activations on the device. This replaced a chunked row-major scheme that re-read
        # all weights per chunk and hit two batched-forward failure modes besides (FLA int32
        # byte-offset overflow past 2^31 bytes of activation, and eager sparse-attention
        # indexers materializing (rows, len, heads, len) fp32 scores over the padded width)
        import threading

        nz = ids != 0
        row_len = (ids.shape[1] - nz.flip(1).int().argmax(dim = 1)) * nz.any(dim = 1).int()
        row_len = row_len.clamp(min = 1).tolist()
        num_rows = ids.shape[0]

        pb_state = {"n": 0}
        pb = ProgressBar("Streaming", len(hook_modules) + 1)

        cv = threading.Condition()
        st = {"active": None, "turn": 0, "error": None}
        tls = threading.local()

        def fail(e):
            if st["error"] is None:
                st["error"] = e
            cv.notify_all()

        def pre_hook(module, args):
            me = id(module)
            load = False
            with cv:
                while True:
                    if st["error"] is not None:
                        raise RuntimeError("layer-major streaming aborted")
                    if st["turn"] == tls.row:
                        if st["active"] == me:
                            return
                        if st["active"] is None:
                            st["active"] = "loading"
                            load = True
                            break
                        if st["active"] != "loading":
                            fail(RuntimeError(
                                f"module sequence diverged between rows: row {tls.row} "
                                f"reached {self.prefix.get(me, '?')} while another module "
                                f"is active (a hooked module is invoked more than once "
                                f"per forward, or rows take different paths)"))
                            raise RuntimeError("layer-major streaming aborted")
                    cv.wait(1.0)
            # materialize outside the lock (heavy I/O); other rows keep waiting on "loading"
            try:
                self._materialize(module)
            except Exception as e:
                with cv:
                    fail(e)
                raise
            with cv:
                st["active"] = me
                cv.notify_all()

        def post_hook(module, args, output):
            with cv:
                st["turn"] += 1
                if st["turn"] == num_rows:
                    self._dematerialize(module)
                    st["active"] = None
                    st["turn"] = 0
                    pb_state["n"] += 1
                    pb.update(pb_state["n"])
                cv.notify_all()

        hooks = []
        # Noise hooks first: they must fire inside the serialized window (before post_hook
        # advances the turn), since their torch.Generator is not thread-safe
        if noise_eps:
            hooks += self._noise_hooks(noise_eps)
        for m in hook_modules:
            hooks.append(m.register_forward_pre_hook(pre_hook))
            hooks.append(m.register_forward_hook(post_hook))

        hidden_cpu = [None] * num_rows

        def worker(r):
            try:
                with torch.inference_mode(), torch.cuda.device(self.device):
                    tls.row = r
                    part = ids[r:r + 1, :row_len[r]].to(self.device)
                    out = base(input_ids = part, use_cache = False).last_hidden_state
                    hidden_cpu[r] = out.to("cpu")
                    del out
            except Exception as e:
                with cv:
                    fail(e)

        try:
            with pb:
                threads = [threading.Thread(target = worker, args = (r,), daemon = True)
                           for r in range(num_rows)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                if st["error"] is not None:
                    raise st["error"]

                self._materialize(head)
                if self.w4a4_sim:
                    n_exp, n_lin = self.a4_count
                    print(f" -- w4a4_sim: {n_exp} expert modules, {n_lin} fp8 linears fake-quantized" +
                          ("" if n_exp or n_lin else "  !! nothing matched the checkpoint's input scales"))
                for r in range(num_rows):
                    logits = head(hidden_cpu[r].to(self.device))
                    if self.logit_multiplier != 1.0:
                        logits = logits * self.logit_multiplier
                    if self.logit_softcap:
                        logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
                    callback(r, logits)
                    hidden_cpu[r] = None
                self._dematerialize(head)
                pb.update(len(hook_modules) + 1)
        finally:
            for h in hooks:
                h.remove()

    @torch.inference_mode()
    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        if self.streaming:
            return self._run_streaming(ids, callback, noise_eps)
        hooks = self._noise_hooks(noise_eps) if noise_eps else []
        try:
            with ProgressBar("Evaluating", ids.shape[0]) as pb:
                for r in range(ids.shape[0]):
                    row = ids[r:r + 1].to(self.model.device)
                    out = self.model(input_ids = row, use_cache = False)
                    callback(r, out.logits)
                    del out
                    pb.update(r + 1)
        finally:
            for h in hooks:
                h.remove()

    def close(self):
        del self.model
        self.shard_handles.clear()
        free_mem()


class _ShardGatherEmbedding(torch.nn.Module):
    """CPU replacement for a sharded-checkpoint nn.Embedding too large to materialize
    (Qwen4Exp's 320M-row n-gram table): rows gather directly from the safetensors mmaps, so
    only the touched pages are ever read. `weight` is a one-row CPU dummy -- the HF caller
    routes ids to weight.device before the lookup and moves the rows back itself."""

    def __init__(self, shards, dtype, scale = None):
        super().__init__()
        self.shards = shards                        # plain list: stays mmap-backed, unregistered
        self.rows_per_shard = shards[0].shape[0]
        self.out_dtype = dtype
        self.scale = scale                          # global dequant scale (fp8-stored tables)
        self.weight = shards[0][:1].float().mul(scale or 1.0).to(dtype)

    def forward(self, ids):
        flat = ids.reshape(-1)
        out = torch.empty((flat.numel(), self.shards[0].shape[1]), dtype = self.out_dtype)
        sh = flat // self.rows_per_shard
        for s in sh.unique().tolist():
            m = sh == s
            rows = self.shards[s][flat[m] - s * self.rows_per_shard].float()
            if self.scale is not None:
                rows = rows * self.scale
            out[m] = rows.to(self.out_dtype)
        return out.view(*ids.shape, -1)


def gguf_shards(source: str) -> list:
    """All files of a split GGUF ("...-00001-of-00003.gguf" naming); [source] when unsplit"""
    import re
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", source)
    if not m:
        return [source]
    base, _, n = m.group(1), m.group(2), int(m.group(3))
    shards = [f"{base}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    missing = [s for s in shards if not os.path.exists(s)]
    assert not missing, f"Split GGUF is missing shards: {missing}"
    return shards


def gguf_storage_info(source: str) -> dict:
    """bpw/vram accounting over the full tensor table, spanning all shards of a split GGUF.
    Norms/biases are < 2 dims; router gates excluded by name; token_embd serves as the head
    fallback for tied models, overridden by output.weight when present in any shard. The PLE
    n-gram table (per_layer_token_embd, Qwen3.8-Flash-Next) is excluded like the embedding:
    the exl3 engine keeps its table out of bpw_layer/vram too (out-of-line file). Hyper-
    connection matrices (hc_*/output_hc_*) and short-conv kernels (*_conv1d) are excluded to
    match the exl3 walk, which counts Linear modules only. Known residual asymmetry: for archs
    whose exl3 routing gate IS a Linear (qwen4exp), exl3 counts it (~0.05% of weights) while
    GGUF_ROUTER_KEYS excludes it here.
    NextN/MTP prediction blocks (packed as the last {arch}.nextn_predict_layers of
    {arch}.block_count) are excluded: the other engines also count decoder layers + head only"""
    from gguf import GGUFReader
    sum_bits = sum_numel = head_bits = head_numel = 0
    head_is_fallback = True
    mtp_prefixes = ()
    for shard_idx, shard in enumerate(gguf_shards(source)):
        reader = GGUFReader(shard)
        if shard_idx == 0:
            def kv(key):
                f = reader.fields.get(key)
                if f is None:
                    return None
                try:
                    return f.contents()
                except Exception:
                    import numpy as np
                    return int(np.array(f.parts[f.data[0]]).flatten()[0])
            arch = kv("general.architecture")
            if arch:
                nextn = kv(f"{arch}.nextn_predict_layers") or 0
                blocks = kv(f"{arch}.block_count")
                if nextn and blocks:
                    mtp_prefixes = tuple(f"blk.{i}." for i in range(int(blocks) - int(nextn), int(blocks)))
        for t in reader.tensors:
            if mtp_prefixes and t.name.startswith(mtp_prefixes):
                continue
            if t.name == "output.weight" or (t.name == "token_embd.weight" and head_is_fallback):
                head_bits = t.n_bytes * 8
                head_numel = t.n_elements
                if t.name == "output.weight":
                    head_is_fallback = False
            elif (
                t.name.endswith(".weight")
                and t.name not in ("token_embd.weight", "per_layer_token_embd.weight")
                and len(t.shape) >= 2
                and not any(k in t.name for k in GGUF_ROUTER_KEYS)
                and ".hc_" not in t.name and not t.name.startswith("output_hc_")
                and not t.name.endswith("_conv1d.weight")
            ):
                sum_bits += t.n_bytes * 8
                sum_numel += t.n_elements
        del reader
    return {
        "bpw_layer": sum_bits / max(sum_numel, 1),
        "bpw_head": head_bits / max(head_numel, 1),
        "vram_gb": (sum_bits + head_bits) / 8 / 1024 ** 3,
    }


class LlamaCppBackend:
    """llama-cpp-python with logits_all; storage info from the GGUF tensor table"""

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        import llama_cpp
        from llama_cpp import Llama
        self.device = device
        self.info = gguf_storage_info(source)
        split_modes = {
            "layer": llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            "row": llama_cpp.LLAMA_SPLIT_MODE_ROW,
            "none": llama_cpp.LLAMA_SPLIT_MODE_NONE,  # everything on main_gpu
        }
        self.model = Llama(
            model_path = source,
            logits_all = True,
            verbose = False,
            n_ctx = max_len,
            n_gpu_layers = options.get("n_gpu_layers", 999),
            split_mode = split_modes[options.get("split_mode", "layer")],
            main_gpu = options.get("main_gpu", 0),
        )
        self.settings = {
            "n_ctx": max_len,
            "n_gpu_layers": options.get("n_gpu_layers", 999),
            "split_mode": options.get("split_mode", "layer"),
            "main_gpu": options.get("main_gpu", 0),
            "llama_cpp": getattr(llama_cpp, "__version__", "?"),
        }

    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        assert not noise_eps, "Noise injection not supported for llamacpp engine"
        with ProgressBar("Evaluating", ids.shape[0]) as pb:
            for r in range(ids.shape[0]):
                self.model.reset()
                self.model.eval(ids[r].tolist())
                logits = torch.from_numpy(self.model.scores).unsqueeze(0)
                logits = logits[:, :ids.shape[1]].to(self.device)
                callback(r, logits)
                pb.update(r + 1)

    def close(self):
        del self.model
        free_mem()


ENGINES = {
    "exllamav3": Exl3Backend,
    "transformers": TransformersBackend,
    "llamacpp": LlamaCppBackend,
}


def open_backend(mspec: dict, max_len: int, device: torch.device):
    engine = mspec["engine"]
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine} (available: {', '.join(ENGINES)})")
    print(f" -- Loading ({engine}): {mspec['source']}")
    options = _Options(mspec.get("options", {}) or {})
    backend = ENGINES[engine](mspec["source"], max_len, device, options)
    # Echo the settings in force (defaults included) so a project can be checked against what
    # the backend actually applied, and flag options no code path read
    for k, v in getattr(backend, "settings", {}).items():
        print(f"      {k}: {v}")
    if options.unused():
        print(f"      ignored options (not used by the {engine} engine): {', '.join(options.unused())}")
    return backend
