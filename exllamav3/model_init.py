from . import Model, Config, Cache, Tokenizer
from .cache import CacheLayer_fp16, CacheLayer_quant
from argparse import ArgumentParser
import torch

def add_args(
    parser: ArgumentParser,
    cache: bool = True
):
    """
    Add standard model loading arguments to command line parser

    :param parser:
        argparse.ArgumentParser

    :param cache:
        bool, include cache arguments. If present, model_init.init() will also return cache
    """
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    parser.add_argument("-gs", "--gpu_split", type = str, help = "Maximum amount of VRAM to use per device, in GB.")
    parser.add_argument("-lm", "--load_metrics", action = "store_true", help = "Show metrics from loader")

    if cache:
        parser.add_argument("-cs", "--cache_size", type = int, help = "Total cache size in tokens, default: 8192", default = 8192)
        parser.add_argument("-cq", "--cache_quant", type = str, help = "Use quantized cache. Specify either kv_bits or k_bits,v_bits pair")

    # TODO:
    # parser.add_argument("-tp", "--tensor_parallel", action = "store_true", help = "Load in tensor-parallel mode")


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

    # Load model
    printp(not quiet, f" -- Loading {args.model_dir}")
    model.load(use_per_device = split, progressbar = progress, **kwargs)

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