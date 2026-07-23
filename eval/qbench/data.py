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
    project["tokenizer"]["source"] = resolve(project["tokenizer"]["source"])
    project["logit_cache"]["dir"] = resolve(project["logit_cache"]["dir"])
    for m in project["models"]:
        m["source"] = resolve(m["source"])
    output = project.get("output", {})
    for key in ("plot_ppl", "plot_kld", "plot_ppl_vram", "plot_kld_vram", "plot_kld_spread",
                "plot_kld_spread_vram", "results", "interactive"):
        if output.get(key):
            output[key] = resolve(output[key])


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
        ids = prepend_hf_chat_context(tokenizer, ids)
        prefix_len = ids.shape[-1] - length

    save_tensors(tokens_file, {"ids": ids, "prefix_len": torch.tensor([prefix_len])})
    return ids, prefix_len


def dataset_subtitle(project: dict) -> str:
    td = project["test_data"]
    name = DATASETS[td["source"].lower()]["display_name"]
    st = f"{name}, {td['rows']} × {td['length']} tokens"
    if project["tokenizer"].get("template"):
        st += ", formatted"
    return st
