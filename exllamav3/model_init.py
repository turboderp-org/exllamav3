from . import Model, Config, Cache, Tokenizer
from .loader import SafetensorsCollection, VariantSafetensorsCollection
from .cache import CacheLayer_fp16, CacheLayer_quant
from argparse import ArgumentParser
import torch
import yaml

def add_args(
    parser: ArgumentParser,
    cache: bool = True,
    default_cache_size = 8192,
):
    """
    Add standard model loading arguments to command line parser

    :param parser:
        argparse.ArgumentParser

    :param cache:
        bool, include cache arguments. If present, model_init.init() will also return cache

    :param default_cache_size:
        Default value for -cs / --cache_size argument
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

    if cache:
        parser.add_argument("-cs", "--cache_size", type = int, help = f"Total cache size in tokens, default: {default_cache_size}", default = default_cache_size)
        parser.add_argument("-cq", "--cache_quant", type = str, help = "Use quantized cache. Specify either kv_bits or k_bits,v_bits pair")


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
        Additional parametes to forwart to Model.load()

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