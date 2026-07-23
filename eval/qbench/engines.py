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

    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        from exllamav3.modules import Linear
        modules = self.model.modules
        states = list(ids.split(1))
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
                    params = {}
                    x = module.prepare_for_device(states[r], params)
                    x = module.forward(x, params)
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
    """

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        from transformers import AutoModelForCausalLM
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32} \
            [options.get("dtype", "bfloat16")]
        self.device = device
        self.dtype = dtype
        self.streaming = options.get("streaming", False)
        self.shard_handles = {}

        if self.streaming:
            import json
            import struct
            from accelerate import init_empty_weights
            from transformers import AutoConfig
            trc = options.get("trust_remote_code", True)
            config = AutoConfig.from_pretrained(source, trust_remote_code = trc)

            # compressed-tensors (llm-compressor) pack-quantized checkpoints are dequantized on
            # the fly by _get_tensor; the skeleton is built unquantized (plain Linear modules),
            # so the quantization config must not reach from_config
            self.ct_bits = None
            qcfg = getattr(config, "quantization_config", None)
            if qcfg is not None:
                if isinstance(qcfg, dict) and qcfg.get("quant_method") == "compressed-tensors":
                    assert qcfg.get("format") == "pack-quantized", \
                        f"Unsupported compressed-tensors format: {qcfg.get('format')}"
                    groups = qcfg.get("config_groups", {})
                    weights = next(iter(groups.values()))["weights"] if groups else {}
                    self.ct_bits = weights.get("num_bits", 4)
                    delattr(config, "quantization_config")
                else:
                    raise ValueError(
                        f"Streaming does not support this quantization_config: "
                        f"{qcfg.get('quant_method') if isinstance(qcfg, dict) else type(qcfg)}"
                    )

            # include_buffers = False keeps non-persistent buffers (rope inv_freq etc., which
            # are not in the shards) materialized with their init values
            with init_empty_weights(include_buffers = False):
                self.model = AutoModelForCausalLM.from_config(config, dtype = dtype, trust_remote_code = trc)
            self.model.eval()

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

            renames = []          # (source substring, target substring)
            self.converters = []  # (target pattern, [source patterns], [ops])
            try:
                from transformers.conversion_mapping import get_checkpoint_conversion_mapping
                for conv in get_checkpoint_conversion_mapping(config.model_type) or []:
                    sources = getattr(conv, "source_patterns", None)
                    targets = getattr(conv, "target_patterns", None)
                    ops = getattr(conv, "operations", None)
                    if not sources or not targets:
                        continue
                    if ops:
                        self.converters.append((targets[0], sources, ops))
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
                    if src.startswith("^") or "(" in src:
                        mod_name = re.sub(src, tgt, mod_name)
                    elif src in mod_name:
                        mod_name = mod_name.replace(src, tgt)
                if mod_name != ck_name:
                    self.tensor_index[mod_name] = (fn, ck_name)

            # Real (init-valued) buffers up front on the compute device
            for m in self.model.modules():
                for k, b in m._buffers.items():
                    if b is not None and not b.is_meta:
                        m._buffers[k] = b.to(device)

            self.prefix = {id(m): n for n, m in self.model.named_modules()}
            # Tied-head fallback: the output embedding's weight is not in the shards when tied
            head = self.model.get_output_embeddings()
            embed = self.model.get_input_embeddings()
            self.head_weight_name = f"{self.prefix[id(head)]}.weight" if head is not None else None
            self.embed_weight_name = f"{self.prefix[id(embed)]}.weight"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype = dtype,
                device_map = options.get("device_map", "auto"),
                trust_remote_code = options.get("trust_remote_code", True),
            )
            self.model.eval()

        # Biases and norms are 1D (excluded by the ndim test); router gates excluded by name.
        # In streaming mode, bits come from the checkpoint's actual stored bytes (including
        # quantization metadata like scales), so quantized formats report their true bitrate
        sum_bits = sum_numel = head_bits = head_numel = 0
        embed_bits = embed_numel = 0
        for name, p in self.model.named_parameters():
            bits = None
            if self.streaming:
                bits = self._stored_bits(name)
            if bits is None:
                bits = p.numel() * p.element_size() * 8
            if "embed" in name:
                embed_bits += bits
                embed_numel += p.numel()
                continue
            if "lm_head" in name or "output" == name.split(".")[0]:
                head_bits += bits
                head_numel += p.numel()
            elif p.ndim >= 2 and not any(k in name for k in HF_ROUTER_KEYS):
                sum_bits += bits
                sum_numel += p.numel()
        if head_numel == 0:
            # Tied embeddings: the head is served by the embedding matrix
            head_bits, head_numel = embed_bits, embed_numel
        self.info = {
            "bpw_layer": sum_bits / max(sum_numel, 1),
            "bpw_head": head_bits / max(head_numel, 1),
            "vram_gb": (sum_bits + head_bits) / 8 / 1024 ** 3,
        }

    def _decoder_layers(self):
        best = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) >= 4:
                if len(set(type(c) for c in m)) == 1 and (best is None or len(m) > len(best)):
                    best = m
        if best is None:
            raise RuntimeError("Could not locate decoder layer list for noise injection")
        return best

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

    CT_SUFFIXES = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx")

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
            if zp.dtype == torch.int32:  # packed like the weights, offset-binary
                zp = ((zp.unsqueeze(-1) >> shifts) & mask).flatten(1)[:, :scale.shape[1]]
                zp = zp - (1 << (bits - 1))
            q = q - zp.repeat_interleave(group, dim = 1)
        return q.to(torch.float32) * scale.repeat_interleave(group, dim = 1)

    def _stored_bits(self, name):
        """Actual stored size in the checkpoint for a module-side tensor name, including any
        quantization metadata; None if the name cannot be resolved"""
        import re
        if name in self.tensor_index:
            return 8 * self._nbytes(name)
        if name.endswith(".weight"):
            stem = name[:-len(".weight")]
            bits = [8 * self._nbytes(f"{stem}.{s}") for s in self.CT_SUFFIXES if f"{stem}.{s}" in self.tensor_index]
            if bits:
                return sum(bits)
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
        return None

    def _get_tensor(self, name):
        import re
        if name in self.tensor_index:
            return self._read_shard(name)
        if name == self.head_weight_name and self.embed_weight_name in self.tensor_index:
            return self._read_shard(self.embed_weight_name)  # tied embeddings

        # compressed-tensors pack-quantized weight
        if name.endswith(".weight"):
            stem = name[:-len(".weight")]
            if f"{stem}.weight_packed" in self.tensor_index:
                return self._dequant_ct(stem)

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

    def _materialize(self, module):
        prefix = self.prefix[id(module)]
        sd = {}
        for pn, _ in module.named_parameters():
            full = f"{prefix}.{pn}" if prefix else pn
            sd[pn] = self._get_tensor(full).to(self.device, self.dtype)
        module.load_state_dict(sd, strict = False, assign = True)
        for pn, p in module.named_parameters():
            if p.is_meta:
                raise RuntimeError(f"No checkpoint tensor for {self.prefix[id(module)]}.{pn}")

    def _dematerialize(self, module):
        for sub in module.modules():
            for pn, p in list(sub._parameters.items()):
                if p is not None and not p.is_meta:
                    sub._parameters[pn] = torch.nn.Parameter(p.to("meta"), requires_grad = False)

    @torch.inference_mode()
    def _run_streaming(self, ids: torch.Tensor, callback, noise_eps: float = None):
        base = self.model.base_model
        layers = self._decoder_layers()
        embed = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()

        # Hook the embedding, every decoder layer, and any other direct child of the base model
        # that carries weights (final norm etc.); the head is streamed manually below
        hook_modules = [embed] + list(layers)
        for child in base.children():
            if child is embed or child is layers:
                continue
            if any(True for _ in child.parameters()):
                hook_modules.append(child)

        pb_state = {"n": 0}
        pb = ProgressBar("Streaming", len(hook_modules) + 1)

        def pre_hook(module, args):
            self._materialize(module)

        def post_hook(module, args, output):
            self._dematerialize(module)
            pb_state["n"] += 1
            pb.update(pb_state["n"])

        hooks = []
        for m in hook_modules:
            hooks.append(m.register_forward_pre_hook(pre_hook))
            hooks.append(m.register_forward_hook(post_hook))
        if noise_eps:
            hooks += self._noise_hooks(noise_eps)

        try:
            with pb:
                # One batched pass: every weight is read exactly once. Activations are
                # rows x len x hidden, tiny next to the weights being streamed
                hidden = base(input_ids = ids.to(self.device), use_cache = False).last_hidden_state
                self._materialize(head)
                for r in range(ids.shape[0]):
                    callback(r, head(hidden[r:r + 1]))
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


class LlamaCppBackend:
    """llama-cpp-python with logits_all; storage info from the GGUF tensor table"""

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        import llama_cpp
        from llama_cpp import Llama
        from gguf import GGUFReader
        self.device = device

        # Norms/biases are < 2 dims; router gates excluded by name; token_embd serves as the
        # head fallback for tied models, overridden by output.weight when present
        reader = GGUFReader(source)
        sum_bits = sum_numel = head_bits = head_numel = 0
        for t in reader.tensors:
            if (t.name == "token_embd.weight" and head_numel == 0) or t.name == "output.weight":
                head_bits = t.n_bytes * 8
                head_numel = t.n_elements
            elif (
                t.name.endswith(".weight")
                and t.name != "token_embd.weight"
                and len(t.shape) >= 2
                and not any(k in t.name for k in GGUF_ROUTER_KEYS)
            ):
                sum_bits += t.n_bytes * 8
                sum_numel += t.n_elements
        self.info = {
            "bpw_layer": sum_bits / max(sum_numel, 1),
            "bpw_head": head_bits / max(head_numel, 1),
            "vram_gb": (sum_bits + head_bits) / 8 / 1024 ** 3,
        }
        del reader
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
    return ENGINES[engine](mspec["source"], max_len, device, mspec.get("options", {}))
