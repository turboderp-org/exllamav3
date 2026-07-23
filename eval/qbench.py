import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import torch
import yaml

"""
qbench: quantization comparison harness driven by a YAML project file.

    python eval/qbench.py project.yml

A project defines a test set, a tokenizer, a logit cache and a list of models. Exactly one model
belongs to the "reference" group; its logits are computed once and cached to disk, then every
other model is streamed through the same test data and compared against the cached logits, so
models never need to be resident at the same time and (for the exllamav3 engine) never need to
be fully resident at all. The reference is also rerun with BF16-rounding-scale noise injected
into the hidden state after every layer, giving the model's self-noise floor: the KLD any lossy
scheme would show even if it introduced no more error than a different-but-equivalent kernel
selection.

KLD is reported as mean/median/p90 over tokens plus buckets by reference confidence. The mean is
dominated by tokens where the reference itself is undecided (near-tie states amplify any
perturbation up to a model-dependent decorrelation floor); the median and the high-confidence
buckets isolate quantization damage from that floor. All results are cached per
(test data, tokenizer, reference, model), so adding one model to a project only runs one model.

Engines: "exllamav3" (module-streamed), "transformers" (full load with accelerate device_map),
"llamacpp" (llama-cpp-python, full logits). A model entry may carry an "options" dict passed to
the engine (e.g. device_map or trust_remote_code for transformers, n_gpu_layers for llamacpp).

Relative paths in the project file resolve against the project file's directory.
"""

from qbench.data import (
    QCache,
    dataset_subtitle,
    get_test_ids,
    resolve_project_paths,
    save_tensors,
    sha_key,
    source_stamp,
)
from qbench.engines import open_backend
from qbench.measure import (
    BF16_ROUNDING_EPS,
    METRICS_VERSION,
    DiffStats,
    print_stats,
    save_reference_row,
)
from qbench.plot import plot_kld_spread, plot_scatter

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)


@torch.inference_mode()
def main(args):
    with open(args.project, "r", encoding = "utf8") as f:
        project = yaml.safe_load(f)
    resolve_project_paths(project, args.project)

    device = torch.device("cuda", args.device)
    cache = QCache(project["logit_cache"])

    ids, prefix_len = get_test_ids(project, cache)
    max_len = ids.shape[-1] + 64
    data_key = sha_key({"v": 1, "test_data": project["test_data"], "tokenizer": project["tokenizer"]})

    from exllamav3 import Config as Exl3Config, Tokenizer as Exl3Tokenizer
    vocab_size = Exl3Tokenizer.from_config(Exl3Config.from_directory(project["tokenizer"]["source"])).actual_vocab_size

    models = project["models"]
    refs = [m for m in models if m.get("group") == "reference"]
    assert len(refs) == 1, f"Project must define exactly one model in the 'reference' group (found {len(refs)})"
    ref = refs[0]

    def model_key(mspec, noise = False):
        # "streaming" is an execution strategy, not a model property; keep it out of the key so
        # toggling it preserves cached logits. Streamed and non-streamed passes are bitwise
        # identical at matching batch shapes; the streamed pass batches all rows, so vs the
        # per-row non-streamed pass the difference is batch-size kernel numerics - the same
        # class of variation as a driver update, well below any model's self-noise floor
        options = {k: v for k, v in mspec.get("options", {}).items() if k != "streaming"}
        return sha_key({
            "v": 1,
            "engine": mspec["engine"],
            "source": mspec["source"],
            "options": options,
            "stamp": source_stamp(mspec["source"]),
            "noise": BF16_ROUNDING_EPS if noise else 0,
        })

    ref_key = model_key(ref)
    ref_store = cache.logits_dir(f"{data_key}_{ref_key}")
    ref_meta = os.path.join(ref_store, "meta.json")

    all_results = []

    # ------ Reference pass: cache logits + confidence, measure ppl
    ref_results_key = f"{data_key}_{ref_key}_self_m{METRICS_VERSION}"
    ref_results = cache.load_results(ref_results_key)
    if ref_results is None or not os.path.exists(ref_meta):
        os.makedirs(ref_store, exist_ok = True)
        backend = open_backend(ref, max_len, device)
        stats = DiffStats(ids, prefix_len, vocab_size, None)
        conf_rows = []
        def ref_callback(r, logits):
            stats(r, logits)
            save_reference_row(ref_store, r, logits, prefix_len, conf_rows)
        backend.run(ids, ref_callback)
        save_tensors(os.path.join(ref_store, "conf.safetensors"), {"conf": torch.cat(conf_rows).half()})
        with open(ref_meta, "w") as f:
            json.dump({"rows": ids.shape[0], "prefix_len": prefix_len}, f)
        ref_results = stats.results()
        ref_results.update(backend.info)
        cache.save_results(ref_results_key, ref_results)
        backend.close()
    print_stats(ref["label"], ref_results)
    all_results.append({"label": ref["label"], "group": ref["group"], **ref_results})

    # ------ Noise floor: reference engine + bf16-rounding noise per layer, vs cached logits
    if project.get("noise_floor", True) and ref["engine"] != "llamacpp":
        floor_results_key = f"{data_key}_{ref_key}_{model_key(ref, noise = True)}_m{METRICS_VERSION}"
        floor_results = cache.load_results(floor_results_key)
        if floor_results is None:
            backend = open_backend(ref, max_len, device)
            stats = DiffStats(ids, prefix_len, vocab_size, ref_store)
            backend.run(ids, stats, noise_eps = BF16_ROUNDING_EPS)
            floor_results = stats.results()
            floor_results.update(backend.info)
            cache.save_results(floor_results_key, floor_results)
            backend.close()
        print_stats("Noise floor", floor_results)
        all_results.append({"label": "Noise floor", "group": "noise_floor", **floor_results})

    # ------ Quantized models
    for mspec in models:
        if mspec is ref:
            continue
        results_key = f"{data_key}_{ref_key}_{model_key(mspec)}_m{METRICS_VERSION}"
        res = cache.load_results(results_key)
        if res is None:
            backend = open_backend(mspec, max_len, device)
            stats = DiffStats(ids, prefix_len, vocab_size, ref_store)
            backend.run(ids, stats)
            res = stats.results()
            res.update(backend.info)
            cache.save_results(results_key, res)
            backend.close()
        print_stats(mspec["label"], res)
        all_results.append({"label": mspec["label"], "group": mspec["group"], **res})

    cache.trim(protect = {ref_store})

    # ------ Output
    output = project.get("output", {})
    if output.get("results"):
        with open(output["results"], "w") as f:
            json.dump(all_results, f, indent = 2)
        print(f" -- Saved results: {output['results']}")

    class PlotArgs:
        title = project.get("title", "qbench")
        subtitle = dataset_subtitle(project)
        vram = False
        max_x = 999999
        max_y = 999999
        dark = output.get("dark", True)
        kld = False
        plot_file = None

    def plot_entries(with_kld, include_ref = True):
        entries = []
        for r in all_results:
            if r["group"] == "noise_floor":
                continue
            if not include_ref and r["group"] == "reference":
                continue
            if "ppl" not in r:  # invalid model (all logits non-finite)
                continue
            if with_kld and "kld" not in r:
                continue
            entries.append({
                "label": f"{r['group']} {r['label']}",
                "layer_bpw": r["bpw_layer"],
                "head_bpw": r["bpw_head"],
                "vram_gb": r["vram_gb"],
                "ppl": r["ppl"],
                **({"kld": r["kld"]} if "kld" in r else {}),
            })
        return entries

    # The reference is a dotted line on ppl charts, not a point: its bpw/vram sits far right of
    # the quants and would compress the interesting region. KLD charts get the noise floor's
    # mean the same way (the scatter's y is the mean KLD, so the mean floor is the comparable
    # line)
    ref_line = {"label": ref["label"], "value": all_results[0]["ppl"]} if "ppl" in all_results[0] else None
    floor_res = next((r for r in all_results if r["group"] == "noise_floor" and "kld" in r), None)
    floor_line = {"label": "noise floor, mean", "value": floor_res["kld"]} if floor_res else None

    def scatter(key, kld, vram):
        if not output.get(key):
            return
        pa = PlotArgs()
        pa.kld = kld
        pa.vram = vram
        pa.plot_file = output[key]
        plot_scatter(
            plot_entries(with_kld = kld, include_ref = kld),
            pa,
            ref_line = floor_line if kld else ref_line,
        )
        print(f" -- Saved plot: {output[key]}")

    scatter("plot_ppl", kld = False, vram = False)
    scatter("plot_kld", kld = True, vram = False)
    scatter("plot_ppl_vram", kld = False, vram = True)
    scatter("plot_kld_vram", kld = True, vram = True)

    for spread_key, spread_vram in (("plot_kld_spread", False), ("plot_kld_spread_vram", True)):
        if output.get(spread_key):
            plot_kld_spread(
                all_results,
                project.get("title", "qbench"),
                dataset_subtitle(project),
                output.get("dark", True),
                output[spread_key],
                caption = output.get("caption", True),
                vram = spread_vram,
            )
            print(f" -- Saved plot: {output[spread_key]}")

    if output.get("interactive"):
        print(" -- output.interactive is not implemented yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("project", type = str, help = "Project file (YAML)")
    parser.add_argument("-d", "--device", type = int, default = 0, help = "Primary CUDA device index")
    main(parser.parse_args())
