from types import SimpleNamespace

from . import Model, Config, Cache, Tokenizer
from .model.config import moe_cpu_layers, moe_cpu_split_sizes
from .util.misc import parse_int_list
from .util.backend import ROCM
from .loader import SafetensorsCollection, VariantSafetensorsCollection
from .cache import CacheLayer_fp16, CacheLayer_quant
from .generator.sampler import ComboSampler
from argparse import ArgumentParser
import re
import torch
import yaml
from pathlib import Path

def add_args(
    parser: ArgumentParser,
    cache: bool = True,
    default_cache_size = 8192,
    default_recurrent_cache_size = 4.0,
    default_cpu_cache_size = 0.0,
    add_sampling_args: bool = False,
    default_sampling_args: dict = None,
    default_autosplit_max_batch_size: int = 1,
    add_draft_model_args: bool = False,
    default_chunk_size: int = 4096,
):
    """
    Add standard model loading arguments to command line parser

    :param parser:
        argparse.ArgumentParser

    :param cache:
        bool, include cache arguments. If present, model_init.init() will also return cache

    :param default_cache_size:
        Default value for -cs / --cache_size argument

    :param default_recurrent_cache_size:
        Default value for -rcs / --recurrent_cache_size argument

    :param default_cpu_cache_size:
        Default value for -ccs / --cpu_cache_size argument

    :param add_sampling_args:
        bool, add sampling arguments

    :param default_autosplit_max_batch_size:
        Default value for -ambs / --autosplit_max_batch_size argument

    :param default_sampling_args:
        dict of default values

    :param add_draft_model_args:
        bool, add draft model args. If True, init() will return draft_model and draft_config as well

    :param default_chunk_size:
        int, value for -chunk_size / --chunk_size
    """
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    parser.add_argument("-gs", "--gpu_split", type = str, help = "Maximum amount of VRAM to use per device, in GB.")
    parser.add_argument("-lpd", "--layers_per_device", type = str, help = "Number of layers to load on each device, example: 2,12,26 (must add up to the model's number of layers, 0 skips a device). Layer-split mode only; --gpu_split still limits the VRAM used per device")
    parser.add_argument("-placement", "--placement", type = str, help = "Where the model's parts go, as text that stands for the other placement arguments: clauses 'subject: setting' separated by ';'. Example: \"layers 0..11: on gpu 0; other layers: on gpu 1; gpu 0: at most 22 GB; gpu 1: at most 22 GB\"")
    parser.add_argument("-lm", "--load_metrics", action = "store_true", help = "Show metrics from loader")
    parser.add_argument("-or", "--override", type = str, help = "Tensor override spec (YAML)", default = None)

    parser.add_argument("-tp", "--tensor_parallel", action = "store_true", help = "Load model in Tensor-parallel mode, attempts to respect --gpu_split")
    parser.add_argument("-mcl", "--moe_cpu_offload", type = moe_cpu_layers, help = "Experimental: run the routed experts of the first N block-sparse MoE layers on the CPU, with expert weights in system RAM. Instead of N, a list of ints or (inclusive) ranges picks the layers by number, example: 8..23 (a single one is written 8..8). Layer-split mode only; requires mul1-codebook experts (ineligible layers fall back to the GPU)", default = 0)
    parser.add_argument("-mcs", "--moe_cpu_split", type = moe_cpu_split_sizes, help = "Experimental: per-layer expert split — run the TAIL N routed experts of every eligible block-sparse MoE layer on the CPU, overlapping the CPU GEMMs with each layer's own GPU expert compute. Instead of N, LAYERS:N items set it per layer, example: 8..23:64,24..39:16. Dynamic hot/cold expert placement is on by default (EXL3_MOE_CPU_SWAP=0 for static placement). Layers taken whole by --moe_cpu_offload are not split. Layer-split mode only; requires mul1-codebook experts", default = 0)
    parser.add_argument("-mct", "--moe_cpu_threads", type = int, help = "Worker thread count for --moe_cpu_offload / --moe_cpu_split (default: EXL3_MOE_CPU_THREADS env, else physical cores minus EXL3_MOE_HOST_CORES)", default = None)
    parser.add_argument("-ngl", "--ngram_lock", action = "store_true", help = "As --ngram_ram, and lock the table's pages in RAM (mlock) so they are never swapped out or reclaimed; needs RLIMIT_MEMLOCK (ulimit -l) to cover the table, or CAP_IPC_LOCK")
    parser.add_argument("-ngr", "--ngram_ram", action = "store_true", help = "Load an n-gram embedding table (PLE models, e.g. Qwen3.8-Flash-Next) fully into system RAM instead of streaming rows from disk per forward (tens of GB of RAM; avoids per-token disk reads)")
    parser.add_argument("-embd", "--embed_disk", action = "store_true", help = "Stream the token embedding table from disk per forward instead of holding it in system RAM")
    parser.add_argument("-tpb", "--tp_backend", type = str, help = "Tensor-parallel backend, either 'native' (default) or 'nccl'", default = "native")
    parser.add_argument("-tp_attn", "--tp_max_parallelism_attn", type = int, help = "(TP) Maximum parallelism for attention layers", default = None)
    parser.add_argument("-tp_mlp", "--tp_max_parallelism_mlp", type = int, help = "(TP) Maximum parallelism for MLP layers", default = None)
    parser.add_argument("-tp_moe", "--tp_max_parallelism_moe", type = int, help = "(TP) Maximum parallelism for MoE layers", default = None)
    parser.add_argument("-tp_linear", "--tp_max_parallelism_linear", type = int, help = "(TP) Maximum parallelism for linear (output) layers", default = None)
    parser.add_argument("-tp_linear_attn", "--tp_max_parallelism_linear_attn", type = int, help = "(TP) Maximum parallelism for linear-attention layers", default = None)
    parser.add_argument("-tp_moe_ts", "--tp_moe_tensor_split", action = "store_true", help = "(TP) Use tensor split for MoE layers rather than expert parallelism")

    parser.add_argument("-swa_full", "--swa_full", action = "store_true", help = f"Use full cache for SWA layers. Default is recurrent mode with snapshots")
    parser.add_argument("-ambs", "--autosplit_max_batch_size", type = int, help = f"Max batch size to account for when loading in autosplit mode (default: {default_autosplit_max_batch_size})", default = default_autosplit_max_batch_size)
    parser.add_argument("-chunk_size", "--chunk_size", type = int, help = f"Max chunk size (default: {default_chunk_size})", default = default_chunk_size)

    parser.add_argument("-lv", "--load_verbose", action = "store_true", help = "Verbose output while loading")
    parser.add_argument("-asnf", "--autosplit_no_forward", action = "store_true", help = "Skip forward pass in autosplit, for debug purposes.")
    parser.add_argument("-nw", "--no_warmup", action = "store_true", help = "Skip the post-load warmup (kernel compilation, GEMM autotuning and graph workspaces are then paid on the first requests instead)")

    parser.add_argument("-layer_map", "--layer_map", type = str, help = "RYS layer map as a list of ints or (inclusive) ranges, example: 0..15,11..31 (repeats layers 11 through 15 once)", default = None)


    if add_sampling_args:
        defs = default_sampling_args if default_sampling_args is not None else {}
        d = SimpleNamespace()
        d.temperature = defs.get("temperature", 0.8)
        d.repetition_penalty = defs.get("repetition_penalty", 1.0)
        d.presence_penalty = defs.get("presence_penalty", 0.0)
        d.frequency_penalty = defs.get("frequency_penalty", 0.0)
        d.penalty_range = defs.get("penalty_range", 1024)
        d.min_p = defs.get("min_p", 0.08)
        d.top_k = defs.get("top_k", 0)
        d.top_p = defs.get("top_p", 1.0)
        d.adaptive_target = defs.get("adaptive_target", 1.0)
        d.adaptive_decay = defs.get("adaptive_decay", 0.9)
        parser.add_argument("-temp", "--temperature", type = float, help = f"Sampling temperature (default: {d.temperature:.1f})", default = d.temperature)
        parser.add_argument("-temp_first", "--temperature_first", action = "store_true", help = "Apply temperature before truncation")
        parser.add_argument("-repp", "--repetition_penalty", type = float, help = f"Repetition penalty, HF style, 1 to disable (default: {d.repetition_penalty:.1f})", default = d.repetition_penalty)
        parser.add_argument("-presp", "--presence_penalty", type = float, help = f"Presence penalty, 0 to disable (default: {d.presence_penalty:.1f})", default = d.presence_penalty)
        parser.add_argument("-freqp", "--frequency_penalty", type = float, help = f"Frequency penalty, 0 to disable (default: {d.frequency_penalty:.1f})", default = d.frequency_penalty)
        parser.add_argument("-penr", "--penalty_range", type = int, help = f"Range for penalties, in tokens (default: {d.penalty_range})", default = d.penalty_range)
        parser.add_argument("-minp", "--min_p", type = float, help = f"Min-P truncation, 0 to disable (default: {d.min_p:.2f})", default = d.min_p)
        parser.add_argument("-topk", "--top_k", type = int, help = f"Top-K truncation, 0 to disable (default: {d.top_k})", default = d.top_k)
        parser.add_argument("-topp", "--top_p", type = float, help = f"Top-P truncation, 1 to disable (default: {d.top_p:.2f})", default = d.top_p)
        parser.add_argument("-adaptive_target", "--adaptive_target", type = float, help = f"Adaptive-P target, 1 to disable (default: {d.adaptive_target:.2f})", default = d.adaptive_target)
        parser.add_argument("-adaptive_decay", "--adaptive_decay", type = float, help = f"Adaptive-P decay, if Adaptive-P enabled (default: {d.adaptive_decay:.2f})", default = d.adaptive_decay)

    if cache:
        parser.add_argument("-cs", "--cache_size", type = int, help = f"Total cache size in tokens, default: {default_cache_size}", default = default_cache_size)
        parser.add_argument("-cq", "--cache_quant", type = str, help = "Use quantized cache. Specify either kv_bits or k_bits,v_bits pair")
        parser.add_argument("-cca", "--cache_compand_a", type = float, help = "Compand a value for simulated cache, default: 0.0", default = 0.0)
        parser.add_argument("-ccs", "--cpu_cache_size", type = float, help = f"CPU second-tier cache size, in GB, default: {default_cpu_cache_size}", default = default_cpu_cache_size)
        parser.add_argument("-rcs", "--recurrent_cache_size", type = float, help = f"CPU second-tier cache size, in GB, default: {default_recurrent_cache_size}", default = default_recurrent_cache_size)

    if add_draft_model_args:
        parser.add_argument("-dm", "--draft_model_dir", type = str, help = "Path to draft model directory", default = None)
        parser.add_argument("-ndt", "--num_draft_tokens", type = int, help = "Number of draft tokens (default: draft model default, else 4)", default = None)
        parser.add_argument("-mtp", "--mtp", action = "store_true", help = "Use MTP drafting")
        parser.add_argument("-ngram", "--ngram_match_min", type = int, help = "N-gram draft minimum match length, default = 0 (disabled)", default = 0)
        parser.add_argument("-ngram_corpus", "--ngram_corpus", type = str, help = "Frozen SAM corpus file for n-gram drafting (requires --ngram_match_min > 0)", default = None)
        parser.add_argument("-dds", "--dynamic_draft", action = "store_true", help = "Dynamically adapt draft length to acceptance rate (num_draft_tokens acts as ceiling)")
        parser.add_argument("-dc", "--draft_confidence", type = float, help = "Confidence target for dynamic draft truncation, default: 0.4", default = 0.4)
        parser.add_argument("-dmcl", "--draft_moe_cpu_layers", type = moe_cpu_layers, help = "Experimental: like --moe_cpu_offload, but for the draft model (or MTP head; a list counts the head's layers from 0)", default = 0)
        parser.add_argument("-dgs", "--draft_gpu_split", type = str, help = "Maximum amount of VRAM to use per device for the draft model (or MTP head), in GB, default: as --gpu_split", default = None)
        parser.add_argument("-dlpd", "--draft_layers_per_device", type = str, help = "As --layers_per_device, for the draft model (or MTP head)", default = None)


def get_arg_sampler(args):
    """
    Create CompoSampler from default args above

    :param args:
        args from ArgumentParser

    :return:
        ComboSampler
    """
    return ComboSampler(
        rep_p = args.repetition_penalty,
        pres_p = args.presence_penalty,
        freq_p = args.frequency_penalty,
        rep_sustain_range = args.penalty_range,
        rep_decay_range = args.penalty_range,
        temperature = args.temperature,
        min_p = args.min_p,
        top_k = args.top_k,
        top_p = args.top_p,
        temp_last = not args.temperature_first,
        adaptive_target = args.adaptive_target,
        adaptive_decay = args.adaptive_decay,
    )


def placement_args(text: str, model) -> tuple[dict, list[str]]:
    """
    Expand a placement description into the arguments it stands for, as they are typed. Clauses are separated by
    ';' or newlines and read 'subject: setting, setting'; their order does not matter:

        layers 0..11: on gpu 0; other layers: on gpu 1      --layers_per_device (every layer, one run per gpu, in order)
        gpu 0: at most 22 GB; gpu 1: unused                 --gpu_split (every gpu or none; per load, as the argument)
        ngram tables: in ram | locked in ram                --ngram_ram | --ngram_lock
        token embedding: on disk; cpu: 16 threads           --embed_disk; --moe_cpu_threads
        layers 20..: all experts on cpu                     --moe_cpu_offload
        layers ..19: 64 experts on cpu                      --moe_cpu_split

    'gpu N' is the N-th GPU the loader lists on this machine, from 0, GPUs only. A clause the text does not know
    is refused, so a word added later changes no text that loads today.
    Returns ({argument: value}, notes).
    """
    def fail(msg):
        raise ValueError(f"Placement: {msg}")

    def runs(layers):
        out = []
        for i in sorted(layers):
            if out and out[-1][1] == i - 1:
                out[-1][1] = i
            else:
                out.append([i, i])
        return ",".join(f"{a}..{b}" for a, b in out)

    if len(text) > 4096:
        fail("the text is longer than 4096 characters")
    num_gpus = torch.cuda.device_count()
    idx = [m.layer_idx for m in model.modules if m.layer_idx is not None and m.layer_idx >= 0]
    n = len(idx)
    if idx != list(range(n)):
        fail("the model's layers are not numbered from 0 in order, use the arguments")
    experts = {
        m.layer_idx: sm.num_experts for m in model.modules if m.layer_idx in idx
        for sm in m if hasattr(sm, "cpu_offload")
    }
    gpu, cpu, limits, other, out, notes = {}, {}, {}, None, {}, []
    for clause in re.split(r"[;\n]", text.lower()):
        clause = " ".join(clause.split())
        if not clause:
            continue
        if clause.count(":") != 1:
            fail(f"'{clause}' needs exactly one ':', as in 'layers 0..11: on gpu 0'")
        subject, settings = (part.strip() for part in clause.split(":"))
        settings = [setting.strip() for setting in settings.split(",")]
        if m := re.fullmatch(r"(?:(all|other) )?layers?(?: (.+))?", subject):
            which, spec = m[1], m[2]
            if bool(which) == bool(spec):
                fail(f"'{subject}': write 'layers 0..11', 'all layers' or 'other layers'")
            layers = list(range(n))
            if spec:
                item = r"[0-9]{1,9}|[0-9]{0,9} ?\.\. ?[0-9]{0,9}"
                if not all(re.fullmatch(item, part.strip()) for part in spec.split(",")):
                    fail(f"'{subject}': layers are numbers and ranges, example: 0..10,12 (a range is 0..10, not 0-10)")
                if bad := [int(i) for i in re.findall(r"[0-9]+", spec) if int(i) >= n]:
                    fail(f"layer {bad[0]} does not exist, the model has layers 0..{n - 1}")
                layers = parse_int_list(spec, min_value = 0, max_value = n - 1)
                if layers != sorted(set(layers)):
                    fail(f"'{subject}': list each layer once, in ascending order")
            for setting in settings:
                if m := re.fullmatch(r"(all|[1-9][0-9]{0,5}) experts? on cpu", setting):
                    if which == "other":
                        fail(f"'{clause}': 'other layers' are the layers no clause puts on a gpu, name the layers here")
                    named = [i for i in layers if i in experts]
                    if (spec or not named) and len(named) < len(layers):
                        rest = runs(set(layers) - set(named))
                        notes.append(f"layers {rest} have no routed experts, '{setting}' does nothing there")
                    for i in named:
                        if i in cpu:
                            fail(f"layer {i} has two experts settings")
                        if m[1] != "all" and int(m[1]) >= experts[i]:
                            fail(f"layer {i} has {experts[i]} routed experts, '{setting}' needs fewer")
                        cpu[i] = m[1]
                    continue
                if not (m := re.fullmatch(r"on gpu ?([0-9]{1,9})", setting)):
                    fail(f"'{setting}' is not valid after '{subject}' (on gpu <n>, all experts on cpu, <n> experts on cpu)")
                if which == "other":
                    if other is not None:
                        fail("'other layers' is given twice")
                    other = int(m[1])
                    continue
                for i in layers:
                    if i in gpu:
                        fail(f"layer {i} is given a gpu twice")
                    gpu[i] = int(m[1])
        elif m := re.fullmatch(r"gpu ?([0-9]{1,9})", subject):
            g, m = int(m[1]), re.fullmatch(r"at most ([0-9]{1,6}(?:\.[0-9]{1,6})?) ?gb|unused", ",".join(settings))
            if not m:
                fail(f"'{clause}': a gpu takes 'at most <n> GB' (GB as in --gpu_split) or 'unused'")
            if g in limits:
                fail(f"gpu {g} has two limits")
            limits[g] = m[1] or "0"
        elif subject == "ngram tables" and settings in (["in ram"], ["locked in ram"]):
            out["ngram_lock" if settings[0][0] == "l" else "ngram_ram"] = True
        elif subject == "token embedding" and settings == ["on disk"]:
            out["embed_disk"] = True
        elif subject == "cpu" and (m := re.fullmatch(r"([1-9][0-9]{0,5}) threads?", ",".join(settings))):
            if out.setdefault("moe_cpu_threads", int(m[1])) != int(m[1]):
                fail("'cpu' is given two thread counts")
        else:
            fail(f"cannot read '{clause}'. Clauses: 'layers 0..11: on gpu 0', 'all layers: on gpu 0', "
                 "'other layers: on gpu 1', 'gpu 0: at most 22 GB', 'gpu 1: unused', 'ngram tables: in ram', "
                 "'ngram tables: locked in ram', 'token embedding: on disk', 'cpu: 16 threads', "
                 "'layers 20..: all experts on cpu', 'layers ..19: 64 experts on cpu'")
    if other is not None:
        if len(gpu) == n:
            fail("'other layers' names no layer, every layer is already on a gpu")
        gpu.update((i, other) for i in range(n) if i not in gpu)
    used = [*gpu.values(), *limits]
    if used and max(used) >= num_gpus:
        fail(f"there is no gpu {max(used)}: {num_gpus} visible, numbered from 0")
    if gpu:
        if len(gpu) < n:
            fail(f"layer {min(set(range(n)) - set(gpu))} is not given a gpu (add 'other layers: on gpu <n>')")
        order = [gpu[i] for i in range(n)]
        if order != sorted(order):
            i = next(i for i in range(1, n) if order[i] < order[i - 1])
            fail(f"layer {i} is on gpu {order[i]} but layer {i - 1} is on gpu {order[i - 1]}: the gpus take the layers in "
                 f"order. For another order, reorder the visible gpus ({'HIP' if ROCM else 'CUDA'}_VISIBLE_DEVICES)")
        out["layers_per_device"] = ",".join(str(order.count(g)) for g in range(order[-1] + 1))
    if limits:
        if len(limits) < num_gpus:
            fail(f"gpu {min(set(range(num_gpus)) - set(limits))} has no limit: with one given, every visible gpu needs "
                 "'at most <n> GB' or 'unused'")
        if bad := [g for g in set(gpu.values()) if not float(limits[g])]:
            fail(f"gpu {bad[0]} is 'unused' but holds layers")
        if not any(float(x) for x in limits.values()):
            fail("every gpu is 'unused'")
        out["gpu_split"] = ",".join(limits[g] for g in range(num_gpus))
    if whole := [i for i in cpu if cpu[i] == "all"]:
        out["moe_cpu_offload"] = runs(whole)
    if split := [
        f"{r}:{k}" for k in dict.fromkeys(cpu.values()) if k != "all"
        for r in runs(i for i in cpu if cpu[i] == k).split(",")
    ]:
        out["moe_cpu_split"] = ",".join(split)
    return out, notes


def init(
    args,
    load_tokenizer: bool = True,
    quiet: bool = False,
    progress: bool = True,
    override_dynamic_seq_len: int | None = None,
    min_draft_len: int = None,
    **kwargs
):
    """
    Create

    :param args:
        argparse.Namespace returned by parse_args()

    :param load_tokenizer:
        bool, also load tokenizer

    :param quiet:
        bool, no console output

    :param progress:
        bool, show rich progress bar while loading

    :param override_dynamic_seq_len:
        (optional) Some models (Like Phi4) have two RoPE modes and adjust their positional embeddings depending on
        sequence length. This argument sets the expected max context length to help select the right mode at load time.
        Mostly relevant if you know ahead of time that you're going to use a long-context model with a short context.

    :param min_draft_len:
        Minimum draft length, even if no draft model is given (for ngram etc.)

    :param kwargs:
        Additional parameters to forwart to Model.load()

    :return:
        tuple of (Model, Config, Cache | None, Tokenizer | None)  or
        tuple of (Model, Config, Cache | None, Tokenizer | None, Model, Config, Cache | None) if draft model args enabled
    """

    def printp(p: bool, s: str):
        if p: print(s)

    return_draft = "draft_model_dir" in args
    draft_model_dir = args.draft_model_dir if return_draft else None
    if "mtp" in args:
        assert not (args.mtp and draft_model_dir), "Cannot specify both --mtp and --draft_model_dir"
        if args.mtp:
            args.draft_model_dir = draft_model_dir = args.model_dir
        use_mtp = draft_model_dir and Path(args.model_dir).resolve() == Path(draft_model_dir).resolve()
    else:
        use_mtp = False

    # Config
    config = Config.from_directory(args.model_dir, layer_map = args.layer_map)
    if override_dynamic_seq_len: config.override_dynamic_seq_len(override_dynamic_seq_len)
    if use_mtp:
        draft_config = config
    elif draft_model_dir:
        draft_config = Config.from_directory(draft_model_dir)
    else:
        draft_config = None

    # Override tensors
    if args.override:
        assert not draft_model_dir, "Tensor overrides not supported when loading with draft model"
        with open(args.override, "r") as f:
            comp = yaml.safe_load(f)
        sources = {s["id"]: s["model_dir"] for s in comp["sources"]}
        overrides = {o["key"]: sources[o["source"]] for o in comp["overrides"]}
        collections = {}
        for o_key, o_dir in overrides.items():
            if o_dir not in collections:
                collections[o_dir] = []
            collections[o_dir].append(o_key)
        if len(collections):
            vstc = VariantSafetensorsCollection(config.stc)
            for o_dir, o_keys in collections.items():
                printp(not quiet, f" -- Overriding from: {o_dir}:")
                for o_key in o_keys:
                    printp(not quiet, f"      {o_key}")
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))
            config.stc = vstc

    # Model instance
    model = Model.from_config(config, swa_full = args.swa_full)
    draft_model = Model.from_config(
        draft_config,
        swa_full = args.swa_full,
        component = "mtp" if use_mtp else "text",
    ) if draft_model_dir else None

    # Cache
    max_history = max(
        min_draft_len or 0,
        draft_model.caps.get("default_draft_size", 4) if draft_model else 0,
        vars(args).get("num_draft_tokens") or 0,
        4 if (vars(args).get("ngram_match_min") and not vars(args).get("num_draft_tokens")) else 0,
    )
    if "cache_size" in vars(args):
        if args.cache_quant is not None:
            split = [int(bits) for bits in args.cache_quant.split(",")]
            if len(split) == 1:
                k_bits = v_bits = split[0]
            elif len(split) == 2:
                k_bits, v_bits = tuple(split)
            else:
                raise ValueError("Specify either one or two bitrates for cache quantization")
            cache = Cache(
                model,
                max_num_tokens = args.cache_size,
                layer_type = CacheLayer_quant,
                k_bits = k_bits,
                v_bits = v_bits,
                compand_a = args.cache_compand_a,
                max_history = max_history,
                max_batch_size = args.autosplit_max_batch_size,
            )
            draft_cache = Cache(
                draft_model,
                max_num_tokens = args.cache_size,
                layer_type = CacheLayer_quant,
                k_bits = k_bits,
                v_bits = v_bits
            ) if draft_model_dir else None
        else:
            cache = Cache(
                model,
                max_num_tokens = args.cache_size,
                layer_type = CacheLayer_fp16,
                max_history = max_history,
                max_batch_size = args.autosplit_max_batch_size,
            )
            draft_cache = Cache(
                draft_model,
                max_num_tokens = args.cache_size,
                layer_type = CacheLayer_fp16
            ) if draft_model_dir else None
    else:
        cache = None
        draft_cache = None

    # Placement
    text = getattr(args, "placement", None)
    if text:
        names = ", ".join(f"gpu {i} = {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count()))
        printp(not quiet and "gpu" in text.lower() and bool(names), f" -- Placement: {names}")
        placed, notes = placement_args(text, model)
        for note in notes:
            printp(not quiet, f" !! Placement: {note}")
        types = {"moe_cpu_offload": moe_cpu_layers, "moe_cpu_split": moe_cpu_split_sizes}
        for name, flag in placed.items():
            given, value = getattr(args, name, None), types.get(name, lambda flag: flag)(flag)
            assert not given or given == value, f"--placement sets --{name} {flag}, and --{name} is given as well: give it once"
            setattr(args, name, value)
        flags = " ".join(f"--{name} {flag}".removesuffix(" True") for name, flag in placed.items())
        printp(not quiet and bool(placed), f" -- Placement: {flags}")

    # Offload
    if getattr(args, "moe_cpu_offload", 0):
        assert not args.tensor_parallel, "--moe_cpu_offload currently requires layer-split mode"
        config.infer_params.moe_cpu_offload = args.moe_cpu_offload
    if getattr(args, "moe_cpu_split", 0):
        assert not args.tensor_parallel, "--moe_cpu_split currently requires layer-split mode"
        config.infer_params.moe_cpu_split = args.moe_cpu_split
    if getattr(args, "moe_cpu_threads", None) is not None:
        config.infer_params.moe_cpu_threads = args.moe_cpu_threads
    if getattr(args, "ngram_ram", False):
        config.infer_params.ngram_stream_from_disk = False
    if getattr(args, "ngram_lock", False):
        config.infer_params.ngram_lock = True
    if getattr(args, "embed_disk", False):
        config.infer_params.embed_stream_from_disk = True
    dmcl = getattr(args, "draft_moe_cpu_layers", 0)
    dmclt = getattr(args, "moe_cpu_threads", None)
    if dmcl:
        assert not args.tensor_parallel, "--draft_moe_cpu_layers currently requires layer-split mode"
        assert draft_model_dir, "--draft_moe_cpu_layers requires a draft model (or --mtp)"
    if getattr(args, "draft_gpu_split", None):
        assert draft_model_dir, "--draft_gpu_split requires a draft model (or --mtp)"
    if getattr(args, "draft_layers_per_device", None):
        assert draft_model_dir, "--draft_layers_per_device requires a draft model (or --mtp)"
    if use_mtp:
        # Shared config: the MTP head is a separate component with its own budget and worker
        config.infer_params.draft_moe_cpu_offload = dmcl
        if dmclt is not None:
            config.infer_params.draft_moe_cpu_threads = dmclt
    elif draft_model_dir:
        # Separate config: the draft model's own text component takes the budget
        draft_config.infer_params.moe_cpu_offload = dmcl
        if dmclt is not None:
            draft_config.infer_params.moe_cpu_threads = dmclt

    # Split
    if args.gpu_split is None or args.gpu_split == "auto":
        split = None
    else:
        split = [float(alloc) for alloc in args.gpu_split.split(",")]
    dgs = getattr(args, "draft_gpu_split", None) or args.gpu_split
    draft_split = None if dgs in (None, "auto") else [float(alloc) for alloc in dgs.split(",")]
    layers = [int(n) for n in args.layers_per_device.split(",")] if getattr(args, "layers_per_device", None) else None
    dlpd = getattr(args, "draft_layers_per_device", None)
    draft_layers = [int(n) for n in dlpd.split(",")] if dlpd else None

    # Parallelism options
    tp_options = {
        "moe_tensor_split": args.tp_moe_tensor_split
    }

    # Parallelism limits
    tp_dev_limits = {}
    for key, arg_name in [
        ("attn", "tp_max_parallelism_attn"),
        ("mlp", "tp_max_parallelism_mlp"),
        ("moe", "tp_max_parallelism_moe"),
        ("linear", "tp_max_parallelism_linear"),
        ("linear_attn", "tp_max_parallelism_linear_attn"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            tp_dev_limits[key] = value
    if len(tp_dev_limits) and not args.tensor_parallel:
        printp(not quiet, " !! Warning, parallelism are do not applied to layer-split model")

    # Load draft model
    if draft_model_dir:
        printp(not quiet, f" -- Loading {draft_model_dir}")
        draft_model.load(
            use_per_device = draft_split,
            layers_per_device = draft_layers,
            progressbar = progress,
            verbose = args.load_verbose,
            max_batch_size = args.autosplit_max_batch_size,
            autosplit_no_forward = args.autosplit_no_forward,
            max_chunk_size = args.chunk_size,
            **kwargs
        )

    # Load model
    printp(not quiet, f" -- Loading {args.model_dir}")
    model.load(
        use_per_device = split,
        layers_per_device = layers,
        tensor_p = args.tensor_parallel,
        progressbar = progress,
        tp_dev_limits = tp_dev_limits,
        tp_backend = args.tp_backend,
        verbose = args.load_verbose,
        tp_options = tp_options,
        max_batch_size = args.autosplit_max_batch_size,
        autosplit_no_forward = args.autosplit_no_forward,
        max_chunk_size = args.chunk_size,
        **kwargs
    )

    # Warmup (before any generator is attached to the cache)
    if not getattr(args, "no_warmup", False):
        printp(not quiet, f" -- Warming up...")
        model.warmup(
            cache = cache,
            max_chunk_size = args.chunk_size,
            progressbar = progress,
            verbose = args.load_verbose,
        )

    # Load tokenizer
    if load_tokenizer:
        printp(not quiet, f" -- Loading tokenizer...")
        tokenizer = Tokenizer.from_config(config)
    else:
        tokenizer = None

    # Metrics
    if args.load_metrics:
        config.stc.metrics.print()

    if return_draft:
        return model, config, cache, tokenizer, draft_model, draft_config, draft_cache
    else:
        return model, config, cache, tokenizer