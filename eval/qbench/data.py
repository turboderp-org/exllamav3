"""
Project spec handling, test data preparation and the central cache.
"""

import hashlib
import json
import os
import shutil

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file

from exllamav3.util.misc import prepend_hf_chat_context

DATASETS = {
    "wiki2": {
        "path": "wikitext", "name": "wikitext-2-raw-v1", "split": "test",
        "text_column": "text", "display_name": "wikitext2",
    },
    "wikitext2": {
        "path": "wikitext", "name": "wikitext-2-raw-v1", "split": "test",
        "text_column": "text", "display_name": "wikitext2",
    },
    "openwebtext10k": {
        "path": "parquet", "name": None, "split": "train",
        "data_files": "hf://datasets/stas/openwebtext-10k@refs/convert/parquet/plain_text/train/*.parquet",
        "text_column": "text", "display_name": "openwebtext",
    },
}


def sha_key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys = True).encode("utf-8")).hexdigest()[:16]


def source_stamp(path: str):
    """
    Modification stamp of a model source, so in-place requantization invalidates caches. Only
    model-content files participate: incidental churn (__pycache__, lock files, readmes) must
    not re-key the reference and silently trigger a full recompute.
    """
    try:
        if os.path.isdir(path):
            files = [
                os.path.join(path, f) for f in os.listdir(path)
                if f.endswith((".safetensors", ".gguf", ".json", ".py"))
            ]
            return max((int(os.path.getmtime(f)) for f in files), default = 0)
        if path.endswith(".gguf"):
            from .engines import gguf_shards
            try:
                return max(int(os.path.getmtime(f)) for f in gguf_shards(path))
            except AssertionError:
                pass  # missing shards surface when the model is opened, not here
        return int(os.path.getmtime(path))
    except OSError:
        return 0


def save_tensors(filename: str, tensors: dict):
    tmp = filename + ".tmp"
    save_file(tensors, tmp)
    os.replace(tmp, filename)


def load_tensor(filename: str, key: str) -> torch.Tensor:
    with safe_open(filename, framework = "pt", device = "cpu") as f:
        return f.get_tensor(key)


def resolve_project_paths(project: dict, project_file: str):
    """Resolve relative paths in the project spec against the project file's directory"""
    base = os.path.dirname(os.path.abspath(project_file))
    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
    if project.get("tokenizer"):
        project["tokenizer"]["source"] = resolve(project["tokenizer"]["source"])
    if project.get("test_trace"):
        project["test_trace"] = resolve(project["test_trace"])
    project["logit_cache"]["dir"] = resolve(project["logit_cache"]["dir"])
    for m in project["models"]:
        m["source"] = resolve(m["source"])
    output = project.get("output", {})
    for key in ("plot_ppl", "plot_kld", "plot_ppl_vram", "plot_kld_vram", "plot_kld_spread",
                "plot_kld_spread_vram", "plot_kld_hist", "plot_kld_hist_combined",
                "results", "interactive"):
        v = output.get(key)
        if not v:
            continue
        # Plot outputs may be a plain path or a dict with a "file" member plus plot options
        # (e.g. plot_kld_hist_combined: {file, x_log, y_log, labels})
        if isinstance(v, dict):
            if v.get("file"):
                v["file"] = resolve(v["file"])
        else:
            output[key] = resolve(v)


class QCache:
    """
    Central cache: tokenized test data, reference logits (one dir of per-row files per
    reference), per-model KLD/ppl results. Only the logit dirs count against max_size_gb and are
    evicted oldest-first.
    """

    def __init__(self, spec: dict):
        self.root = os.path.join(spec["dir"], "qbench")
        self.max_size = int(spec.get("max_size_gb", 200) * 1024 ** 3)
        os.makedirs(self.root, exist_ok = True)

    def tokens_file(self, key):
        return os.path.join(self.root, f"tokens_{key}.safetensors")

    def logits_dir(self, key):
        return os.path.join(self.root, f"logits_{key}")

    def results_file(self, key):
        return os.path.join(self.root, f"results_{key}.json")

    def load_results(self, key):
        try:
            with open(self.results_file(key), "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def save_results(self, key, results: dict):
        with open(self.results_file(key), "w") as f:
            json.dump(results, f, indent = 2)

    # Per-token KLD sidecar next to each results JSON (fp32, ~4 bytes/token): lets the
    # histogram outputs pair tokens across passes ((model KLD - floor KLD) per token) without
    # keeping full logits around. fp32 rather than fp16: sub-6e-8 KLDs (common on in-domain
    # trace data, where quants often reproduce the reference near-exactly) flush to zero in
    # fp16, censoring the left tail of the distribution. Not counted against max_size_gb
    def kl_file(self, key):
        return os.path.join(self.root, f"kl_{key}.safetensors")

    def save_kl(self, key, kl: torch.Tensor | None):
        if kl is not None:
            save_tensors(self.kl_file(key), {"kl": kl.float().cpu()})

    def load_kl(self, key) -> torch.Tensor | None:
        if not os.path.exists(self.kl_file(key)):
            return None
        try:
            return load_tensor(self.kl_file(key), "kl")
        except Exception:
            return None

    def trim(self, protect: set):
        dirs = []
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            if not name.startswith("logits_") or not os.path.isdir(path) or path in protect:
                continue
            size = sum(
                os.path.getsize(os.path.join(path, f))
                for f in os.listdir(path)
            )
            dirs.append((os.path.getmtime(path), path, size))
        protected_size = sum(
            os.path.getsize(os.path.join(p, f))
            for p in protect if os.path.isdir(p)
            for f in os.listdir(p)
        )
        total = protected_size + sum(d[2] for d in dirs)
        dirs.sort()
        while total > self.max_size and dirs:
            _, path, size = dirs.pop(0)
            print(f" -- Cache limit: evicting {path} ({size / 1024**3:.1f} GB)")
            shutil.rmtree(path)
            total -= size


def get_test_rows(project: dict, cache: QCache):
    """
    Test rows as (ids [rows, max_len] right-padded, ranges, vocab_size_or_None): per row,
    metrics run over logits positions [a, b). From `test_trace` (a qbench_prompts.py trace:
    variable-length (context, response) pairs, scored only on the sampled response positions -
    padding beyond b is causally inert), or from the classic test_data/tokenizer spec (uniform
    rows, scored from the chat-context prefix onward).
    """
    if project.get("test_trace"):
        with open(project["test_trace"], "r") as f:
            trace = json.load(f)
        rows = trace["rows"]
        assert rows, "test_trace contains no rows"
        max_len = max(len(r["input_ids"]) + len(r["response_ids"]) for r in rows)
        ids = torch.zeros((len(rows), max_len), dtype = torch.long)
        ranges = []
        for i, r in enumerate(rows):
            seq = r["input_ids"] + r["response_ids"]
            ids[i, :len(seq)] = torch.tensor(seq, dtype = torch.long)
            p = len(r["input_ids"])
            # logits at [p-1, p+R) predict exactly the R sampled tokens (plus the final
            # next-token distribution, mirroring the classic mode's inclusive end)
            ranges.append((p - 1, len(seq)))
        return ids, ranges, trace.get("vocab_size")

    ids, prefix_len = get_test_ids(project, cache)
    return ids, [(prefix_len, ids.shape[-1])] * ids.shape[0], None


def get_test_ids(project: dict, cache: QCache):
    """
    Tokenized test rows, cached by (dataset spec, tokenizer, template flag). Returns
    (ids [rows, total_len], prefix_len); all metrics are computed on positions >= prefix_len.
    """
    td = project["test_data"]
    tok = project["tokenizer"]
    key = sha_key({"v": 1, "test_data": td, "tokenizer": tok})
    tokens_file = cache.tokens_file(key)

    if os.path.exists(tokens_file):
        ids = load_tensor(tokens_file, "ids")
        prefix_len = int(load_tensor(tokens_file, "prefix_len").item())
        return ids, prefix_len

    from exllamav3 import Tokenizer, Config
    spec = DATASETS[td["source"].lower()]
    print(f" -- Loading text dataset: {spec['path']}" + (f"/{spec['name']}" if spec["name"] else ""))
    if spec["name"] is None:
        ds = load_dataset(spec["path"], split = spec["split"], data_files = spec.get("data_files"))
    else:
        ds = load_dataset(spec["path"], spec["name"], split = spec["split"], data_files = spec.get("data_files"))
    text = "\n\n".join(t for t in ds[spec["text_column"]] if isinstance(t, str) and t.strip())

    config = Config.from_directory(tok["source"])
    tokenizer = Tokenizer.from_config(config)
    print(f" -- Tokenizing")
    tokens = tokenizer.encode(text)
    rows, length, stride = td["rows"], td["length"], td.get("stride", td["length"])
    seqs = []
    for a in range(0, tokens.shape[-1] - length, stride):
        seqs.append(tokens[:, a:a + length])
        if len(seqs) >= rows:
            break
    if len(seqs) < rows:
        raise ValueError(f"Dataset only provides {len(seqs)} rows of {length} tokens")
    ids = torch.cat(seqs, dim = 0)

    prefix_len = 0
    if tok.get("template"):
        # template: true wraps rows after a bare generation prompt; template: assistant embeds
        # them as an unterminated assistant message (in-distribution for structured formats
        # like gpt-oss harmony, equivalent otherwise)
        mode = "assistant" if tok.get("template") == "assistant" else "generation"
        ids = prepend_hf_chat_context(tokenizer, ids, mode = mode,
                                      prompt = tok.get("prompt", "Say something."))
        prefix_len = ids.shape[-1] - length

    save_tensors(tokens_file, {"ids": ids, "prefix_len": torch.tensor([prefix_len])})
    return ids, prefix_len


def dataset_subtitle(project: dict) -> str:
    if project.get("test_trace"):
        with open(project["test_trace"], "r") as f:
            meta = json.load(f).get("meta", {})
        return (f"self-generated in-domain trace, {meta.get('input_tokens', 0):,} input + "
                f"{meta.get('output_tokens', 0):,} output tokens")
    td = project["test_data"]
    name = DATASETS[td["source"].lower()]["display_name"]
    st = f"{name}, {td['rows']} × {td['length']} tokens"
    if project["tokenizer"].get("template") == "assistant":
        st += ", assistant-framed"
    elif project["tokenizer"].get("template"):
        st += ", formatted"
    return st
