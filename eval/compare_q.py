import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from datasets import load_dataset
import math
import argparse
import json
import glob
from safetensors.torch import save_file
from safetensors import safe_open
import gc
try:
    from compare_q_plot import plot
except ImportError:
    from eval.compare_q_plot import plot

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

# Lookup tables to ensure test functions are cacheable

from compare_q_transformers import (
    load_transformers_auto_bf16,
    load_transformers_auto,
    load_transformers,
    fwd_transformers,
    tokenize_transformers,
    chat_template_transformers,
    load_transformers_mm
)
from compare_q_exllamav2 import (
    load_exllamav2,
    fwd_exllamav2
)
from compare_q_exllamav3 import (
    load_exllamav3,
    fwd_exllamav3
)
from compare_q_llamacpp import (
    load_llamacpp,
    fwd_llamacpp
)
from compare_q_anyprecision import (
    load_anyprecision,
    fwd_anyprecision,
)
from compare_q_qtip import (
    load_qtip,
    fwd_qtip,
)

load_fns = {
    "transformers_auto_bf16": load_transformers_auto_bf16,
    "transformers_auto": load_transformers_auto,
    "transformers_mm": load_transformers_mm,
    "transformers": load_transformers,
    "exllamav2": load_exllamav2,
    "exllamav3": load_exllamav3,
    "llamacpp": load_llamacpp,
    "anyprecision": load_anyprecision,
    "qtip": load_qtip,
}

fwd_fns = {
    "transformers": fwd_transformers,
    "exllamav2": fwd_exllamav2,
    "exllamav3": fwd_exllamav3,
    "llamacpp": fwd_llamacpp,
    "anyprecision": fwd_anyprecision,
    "qtip": fwd_qtip,
}

tokenize_fns = {
    "transformers": tokenize_transformers,
}

template_fns = {
    "transformers": chat_template_transformers,
}

# Util fn

def load_tensor(filename):
    with safe_open(filename, framework = "pt", device = "cpu") as f:
        if "tensor" in f.keys():
            return f.get_tensor("tensor")
        else:
            tensors = []
            i = 0
            while f"tensor.{i}" in f.keys():
                tensors.append(f.get_tensor(f"tensor.{i}"))
                i += 1
            return tensors

def save_tensor(tensor, filename: str):
    if isinstance(tensor, dict):
        save_file({k: v for k, v in tensor.items()}, filename)
    elif isinstance(tensor, list):
        save_file({f"tensor.{i}": t for i, t in enumerate(tensor)}, filename)
    else:
        save_file({f"tensor": tensor}, filename)

# Tokenize ppl test data

DATASET_ALIASES = {
    "wiki2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
    },
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
    },
    "wiki103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "text_column": "text",
    },
    "wikitext103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "text_column": "text",
    },
    "ptb": {
        "path": "ptb_text_only",
        "name": "penn_treebank",
        "split": "test",
        "text_column": "sentence",
    },
    "lambada": {
        "path": "EleutherAI/lambada_openai",
        "name": None,
        "split": "test",
        "text_column": "text",
    },
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "split": "validation",
        "text_column": "text",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "text_column": "text",
    },
    "openwebtext10k": {
        "path": "stas/openwebtext-10k",
        "name": None,
        "split": "val",
        "text_column": "text",
    },
    "openwebtext": {
        "path": "Skylion007/openwebtext",
        "name": None,
        "split": "train[:1000]",
        "text_column": "text",
    },
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train[:1000]",
        "text_column": "text",
    },
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train[:1000]",
        "text_column": "text",
    },
}


def get_dataset_text(spec: dict) -> str:
    dataset = spec["dataset"]
    dataset_spec = DATASET_ALIASES.get(dataset.lower(), {})
    path = spec.get("dataset_path", dataset_spec.get("path", dataset))
    name = spec.get("dataset_name", dataset_spec.get("name"))
    split = spec.get("dataset_split", spec.get("split", dataset_spec.get("split", "test")))
    text_column = spec.get("text_column", dataset_spec.get("text_column", "text"))
    max_text_rows = spec.get("max_text_rows", spec.get("max_dataset_rows", 0))

    print(f"Loading text dataset: {path}" + (f"/{name}" if name else "") + f" ({split})")
    if name is None:
        ds = load_dataset(path, split = split)
    else:
        ds = load_dataset(path, name, split = split)

    if text_column not in ds.column_names:
        raise ValueError(
            f"Dataset '{dataset}' does not have text column '{text_column}'. "
            f"Available columns: {', '.join(ds.column_names)}"
        )

    texts = ds[text_column]
    if max_text_rows:
        texts = texts[:max_text_rows]
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError(f"Dataset '{dataset}' produced no non-empty text rows")
    return "\n\n".join(texts)


@disk_lru_cache("get_dataset")
def get_test_data(spec: dict):
    tokenize_fn = tokenize_fns[spec["tokenize_fn"]]
    template_fn = template_fns[spec["tokenize_fn"]] if spec.get("chat_template") else None
    eval_stride = spec["eval_stride"]
    eval_len = spec["eval_len"]
    max_rows = spec.get("max_rows", 0)
    eval_tokens = tokenize_fn(
        spec["tokenizer_dir"],
        get_dataset_text(spec)
    )
    num_tokens = eval_tokens.shape[-1]
    seqs = []
    for a in range(0, num_tokens - eval_len, eval_stride):
        b = a + eval_len
        tokens = eval_tokens[:, a:b]
        if template_fn:
            tokens = template_fn(spec["tokenizer_dir"], tokens)
        seqs.append(tokens)
        if max_rows and len(seqs) >= max_rows:
            break
    eval_tokens = torch.cat(seqs, dim = 0)[:, :]
    return eval_tokens

# Run ppl test

@disk_lru_cache("test_ppl")
def test_ppl(data_spec: dict, spec: dict, logits_file: str):
    load_fn = load_fns[spec["load_fn"]]
    fwd_fn = fwd_fns[spec["fwd_fn"]]
    model_dir = spec["model_dir"]

    print(f"Loading dataset: {data_spec['dataset']}")
    eval_ids = get_test_data(data_spec)
    rows = eval_ids.shape[0]
    length = eval_ids.shape[1]

    print(f"Loading: {model_dir}")
    model_instance, bpw_layer, bpw_head, vram_bits = load_fn(model_dir, size = length + 512)
    vram_gb = vram_bits / 8 / 1024**3

    logprob_sum = 0.0
    logprob_count = 0
    kl_div_sum_ab = 0.0
    kl_div_count = 0.0

    eval_len = data_spec["eval_len"] - data_spec.get("warmup_tokens", 0)

    print(f"Testing: {model_dir} ({spec['label']})")

    collect_logits = False
    if logits_file:
        if "out_logits" in spec:
            collect_logits = True
            ref_logits = []
        else:
            collect_logits = False
            ref_logits = load_tensor(logits_file)
            if not isinstance(ref_logits, list):
                ref_logits = ref_logits.split(1, 0)

    with ProgressBar("Evaluating", rows) as pb:
        for row in range(rows):
            pb.update(row)
            input_ids = eval_ids[row:row + 1, :]
            logits = fwd_fn(model_instance, input_ids)
            logits = logits.float()[..., -eval_len:, :]

            # kld
            if logits_file and row < 10:
                probs_a = torch.softmax(logits, dim = -1)
                if collect_logits:
                    ref_logits.append(logits.cpu())
                    kl_div_count += 1
                else:
                    probs_b = torch.softmax(ref_logits[row].to(logits.device), dim = -1)
                    vs = min(probs_a.shape[-1], probs_b.shape[-1])
                    probs_a = probs_a[..., :vs]
                    probs_b = probs_b[..., :vs]
                    for r in range(probs_a.shape[1]):
                        kl_div = F.kl_div(torch.log(probs_a[:, r:r+1, :] + 1e-10), probs_b[:, r:r+1, :], reduction = 'sum')
                        kl_div_sum_ab += kl_div.item()
                        kl_div_count += 1
                    del kl_div
                    del probs_b
                del probs_a

            # ppl
            logits = logits[:, :-1, :]
            logits += 1e-10
            log_probs = F.log_softmax(logits, dim = -1)
            del logits
            target_ids = input_ids[:, -eval_len:][:, 1:].to(log_probs.device)
            del input_ids
            target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            del log_probs
            logprob_sum += target_log_probs.sum().item()
            logprob_count += target_ids.numel()
            del target_log_probs
            del target_ids
            torch.cuda.empty_cache()

        pb.update(rows)

    mean_log_prob = logprob_sum / logprob_count
    perplexity = math.exp(-mean_log_prob)
    if logits_file:
        kl_div = kl_div_sum_ab / kl_div_count
        print(f"KL div: {kl_div:.6f}")

    if collect_logits:
        save_tensor(ref_logits, logits_file)

    print(f"Perplexity: {perplexity:.6f}")

    del model_instance
    del eval_ids

    free_mem()
    res = {
        "label": spec.get("label", spec.get("model_dir")),
        "layer_bpw": bpw_layer,
        "head_bpw": bpw_head,
        "vram_gb": vram_gb,
        "ppl": perplexity
    }
    if logits_file:
        res.update({
            "kld": kl_div
        })

    return res


def dict_hash(x: dict) -> str:
    import hashlib
    key = str(json.dumps(x, sort_keys = True))
    encoded_string = key.encode('utf-8')
    hash_object = hashlib.sha256(encoded_string)
    hex_digest = hash_object.hexdigest()
    return hex_digest


@torch.inference_mode()
def main(args):
    with open(args.dataspec, "r", encoding = "utf8") as f:
        test_data_spec = json.load(f)

    models_files = args.modelspec
    models_files_g = []
    models_spec = []
    for filename in models_files:
        if "*" in filename:
            models_files_g += glob.glob(filename)
        else:
            models_files_g.append(filename)
    for filename in models_files_g:
        with open(filename, "r", encoding = "utf8") as f:
            m = json.load(f)
            models_spec += m

    if args.logits_file:
        logits_file = args.logits_file
    else:
        logits_file = None
        for idx, spec in enumerate(models_spec):
            if "out_logits" in spec:
                logits_dir = spec["out_logits"]
                if not os.path.exists(logits_dir):
                    os.makedirs(logits_dir)
                logits_file = os.path.join(logits_dir, dict_hash(test_data_spec) + ".safetensors")
                logits_idx = idx
        if logits_file is not None:
            models_spec = [models_spec[logits_idx]] + models_spec[:logits_idx] + models_spec[logits_idx + 1:]

    if args.mask:
        masks = args.mask.split(";")
        ms = []
        for spec in models_spec:
            if any(m.upper() in spec["label"].upper() for m in masks):
                ms.append(spec)
        models_spec = ms

    if args.clear_cache:
        for spec in models_spec:
            disk_lru_cache_clear("test_ppl", test_data_spec, spec, logits_file)

    results = []
    for spec in models_spec:
        r = test_ppl(test_data_spec, spec, logits_file)
        print(r)
        results.append(r)

        torch.cuda.empty_cache()
        gc.collect()

    print("------")
    print(json.dumps(results, indent = 4))

    if args.plot:
        plot(results, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataspec", type = str, help = "Data specification (JSON file)")
    parser.add_argument("-m", "--modelspec", type = str, nargs="+", help = "Model specification (JSONL file), accepts wildcard")
    parser.add_argument("-cc", "--clear_cache", action = "store_true", help = "Clear cache")
    parser.add_argument("-p", "--plot", action = "store_true", help = "Scatter plot")
    parser.add_argument("-v", "--vram", action = "store_true", help = "Use VRAM footprint as scatter plot X axis")
    parser.add_argument("-mx", "--max_x", type = float, default = 999999, help = "Don't plot results beyond X value")
    parser.add_argument("-my", "--max_y", type = float, default = 999999, help = "Don't plot results beyond Y value")
    parser.add_argument("-t", "--title", type = str, default = "Very plot", help = "Plot title")
    parser.add_argument("-kld", "--kld", action = "store_true", help = "Test KL divergence")
    parser.add_argument("-mask", "--mask", type = str, help = "Semicolon-separated list of strings to match against model labels for inclusion")
    parser.add_argument("-lf", "--logits_file", type = str, help = "Reference logits file for KLD", required = False)
    parser.add_argument("-dark", "--dark", action = "store_true", help = "Dark mode")
    parser.add_argument("-pf", "--plot_file", type = str, help = "Write the plot to a file")
    _args = parser.parse_args()
    main(_args)
