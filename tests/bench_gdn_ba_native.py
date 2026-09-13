"""Fixed 3000/1000 native Model-API workload for a base/candidate build comparison."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Cache, Config, Model
from exllamav3.ext import exllamav3_ext as ext


def target_layers(model):
    layers = [block.attn for block in model.modules
        if getattr(getattr(block, "attn", None), "bc_split", False)
        and not getattr(block.attn, "kda", False)
        and tuple(block.attn.ba_weight_t.shape) == (96, 5120)]
    if not layers:
        raise ValueError("checkpoint has no native split-GDN B/A K5120/N96 caller")
    return layers


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def generate(model, cache, prompt, steps):
    states = None
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(prompt) - 1, 256):
            params = dict(attn_mode = "flash_attn", cache = cache, past_len = offset, batch_shape = (1, 4096))
            if states is not None:
                params["recurrent_states"] = states
            ids = torch.tensor([prompt[offset:min(offset + 256, len(prompt) - 1)]], dtype = torch.long)
            model.prefill(ids, params)
            states = params.get("recurrent_states")
        torch.cuda.synchronize()
        prefill = time.perf_counter() - start
        x = torch.tensor([[prompt[-1]]], dtype = torch.long)
        tokens, finite = [], []
        for step in range(steps):
            logits = model.forward(x, dict(attn_mode = "flash_attn", cache = cache,
                past_len = len(prompt) - 1 + step, batch_shape = (1, 4096), recurrent_states = states))
            token = int(logits.argmax(-1).item())
            tokens.append(token)
            if step == 0:
                ttft = time.perf_counter() - start
            if step % 128 == 0 or step == steps - 1:
                finite.append(torch.isfinite(logits).all())
            x = torch.tensor([[token]], dtype = torch.long)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert bool(torch.stack(finite).all())
        return dict(elapsed_ms = elapsed * 1000, prefill_ms = prefill * 1000,
            ttft_ms = ttft * 1000, tpot_ms = (elapsed - ttft) * 1000 / (steps - 1),
            output_tokens = len(tokens), tokens = tokens, finite = True)
    finally:
        if states:
            for state in states:
                state.free()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", required = True)
    parser.add_argument("--label", required = True, help = "Identify the installed base or candidate build")
    parser.add_argument("--output", type = Path, required = True)
    parser.add_argument("--source-revision", required = True, help = "Frozen source commit, including for an archive build")
    parser.add_argument("--checkpoint-revision", required = True, help = "Immutable checkpoint repository revision")
    parser.add_argument("--case", type = int, choices = range(3), help = "Run one of the three frozen prompt cases")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    config = Config.from_directory(args.model)
    raw_config = json.loads((Path(args.model) / "config.json").read_text())
    vocabulary = raw_config.get("text_config", raw_config).get("vocab_size", 0)
    if vocabulary < 240000:
        raise ValueError("the frozen random-token workload requires vocab_size >= 240000")
    manifest = Path(args.model) / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError("provide the checkpoint SHA256SUMS manifest and verify it before measurement")
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    rng = random.Random(20260904)
    prompts = [[rng.randrange(1000, 240000) for _ in range(3000)] for _ in range(3)]
    model.load(device = "cuda:0", progressbar = False)
    try:
        layers = target_layers(model)
        generate(model, cache, prompts[0], 8)
        runs = []
        for index, prompt in enumerate(prompts):
            if args.case is not None and index != args.case:
                continue
            start_epoch = time.time()
            row = generate(model, cache, prompt, 1000)
            row.update(start_epoch = start_epoch, end_epoch = time.time())
            row.update(case = index, prompt_sha256 = hashlib.sha256(json.dumps(prompt, separators = (",", ":")).encode()).hexdigest())
            runs.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush = True)
        root = Path(__file__).resolve().parents[1]
        data = dict(label = args.label, scope = "native Model API, batch1, fixed random-token input,3000/1000,FP16 KV,prefill chunk256",
            seed = 20260904, model_config_sha256 = digest(Path(args.model) / "config.json"),
            extension_sha256 = digest(ext.__file__), gdn_source_sha256 = digest(root / "exllamav3/exllamav3_ext/gdn.cu"),
            source_revision = args.source_revision, checkpoint_revision = args.checkpoint_revision,
            checkpoint_manifest_sha256 = digest(manifest), target_layer_count = len(layers),
            exllamav3 = importlib.metadata.version("exllamav3"), torch = torch.__version__, cuda = torch.version.cuda,
            gpu = torch.cuda.get_device_name(0), capability = torch.cuda.get_device_capability(0), runs = runs)
        args.output.write_text(json.dumps(data, indent = 2) + "\n")
    finally:
        model.unload()


if __name__ == "__main__":
    main()
