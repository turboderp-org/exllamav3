from types import SimpleNamespace

from . import Model, Config, Cache, Tokenizer
from .loader import SafetensorsCollection, VariantSafetensorsCollection
from .cache import CacheLayer_fp16, CacheLayer_quant
from .generator.sampler import ComboSampler
from argparse import ArgumentParser
import yaml

def add_args(
    parser: ArgumentParser,
    cache: bool = True,
    default_cache_size = 8192,
    add_sampling_args: bool = False,
    default_sampling_args: dict = None
):
    """
    Add standard model loading arguments to command line parser

    :param parser:
        argparse.ArgumentParser

    :param cache:
        bool, include cache arguments. If present, model_init.init() will also return cache

    :param default_cache_size:
        Default value for -cs / --cache_size argument

    :param add_sampling_args:
        bool, add sampling arguments

    :param default_sampling_args:
        dict of default values
    """
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    parser.add_argument("-gs", "--gpu_split", type = str, help = "Maximum amount of VRAM to use per device, in GB.")
    parser.add_argument("-lm", "--load_metrics", action = "store_true", help = "Show metrics from loader")
    parser.add_argument("-or", "--override", type = str, help = "Tensor override spec (YAML)", default = None)

    parser.add_argument("-tp", "--tensor_parallel", action = "store_true", help = "Load model in Tensor-parallel mode, attempts to respect --gpu_split")
    parser.add_argument("-tpb", "--tp_backend", type = str, help = "Tensor-parallel backend, either 'native' (default) or 'nccl'", default = "native")
    parser.add_argument("-tp_attn", "--tp_max_parallelism_attn", type = int, help = "(TP) Maximum parallelism for attention layers", default = None)
    parser.add_argument("-tp_mlp", "--tp_max_parallelism_mlp", type = int, help = "(TP) Maximum parallelism for MLP layers", default = None)
    parser.add_argument("-tp_moe", "--tp_max_parallelism_moe", type = int, help = "(TP) Maximum parallelism for MoE layers", default = None)
    parser.add_argument("-tp_linear", "--tp_max_parallelism_linear", type = int, help = "(TP) Maximum parallelism for linear (output) layers", default = None)
    parser.add_argument("-tp_moe_ts", "--tp_moe_tensor_split", action = "store_true", help = "(TP) Use tensor split for MoE layers rather than expert parallelism")

    parser.add_argument("-lv", "--load_verbose", action = "store_true", help = "Verbose output while loading")

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


def init(
    args,
    load_tokenizer: bool = True,
    quiet: bool = False,
    progress: bool = True,
    override_dynamic_seq_len: int | None = None,
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

    :param kwargs:
        Additional parameters to forwart to Model.load()

    :return:
        tuple of (Model, Config, Cache | None, Tokenizer | None)
    """

    def printp(p: bool, s: str):
        if p: print(s)

    # Config
    config = Config.from_directory(args.model_dir)
    if override_dynamic_seq_len: config.override_dynamic_seq_len(override_dynamic_seq_len)

    # Override tensors
    if args.override:
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
    model = Model.from_config(config)

    # Cache
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
                v_bits = v_bits
            )
        else:
            cache = Cache(
                model,
                max_num_tokens = args.cache_size,
                layer_type = CacheLayer_fp16
            )
    else:
        cache = None

    # Split
    if args.gpu_split is None or args.gpu_split == "auto":
        split = None
    else:
        split = [float(alloc) for alloc in args.gpu_split.split(",")]

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
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            tp_dev_limits[key] = value
    if len(tp_dev_limits) and not args.tensor_parallel:
        printp(not quiet, " !! Warning, parallelism are do not applied to layer-split model")

    # Load model
    printp(not quiet, f" -- Loading {args.model_dir}")
    model.load(
        use_per_device = split,
        tensor_p = args.tensor_parallel,
        progressbar = progress,
        tp_dev_limits = tp_dev_limits,
        tp_backend = args.tp_backend,
        verbose = args.load_verbose,
        tp_options = tp_options,
        **kwargs
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

    return model, config, cache, tokenizer