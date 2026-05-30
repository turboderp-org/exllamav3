import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model
import argparse
from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection
import yaml
from tabulate import tabulate

def print_markdown_table(stats):
    """Print layer statistics as a pretty-aligned Markdown table using tabulate."""
    total_bytes = sum(s["size_bytes"] for s in stats.values())

    # Sort by size descending
    sorted_layers = sorted(stats.items(), key=lambda x: -x[1]["size_bytes"])

    # Build table rows with raw numeric values
    rows = []
    for layer_name, stat in sorted_layers:
        size_mib = stat["size_bytes"] / 1024**2
        size_pct = 100 * stat["size_bytes"] / total_bytes if total_bytes > 0 else 0
        bpw = stat["bits"] / stat["numel"]

        rows.append([layer_name, stat["count"], size_mib, size_pct, bpw])

    # Print table with github format
    print()
    print(
        tabulate(
            rows,
            headers=["Layer name", "Number", "Size (MiB)", "Size (%)", "Effective BPW"],
            tablefmt="github",
            stralign="left",
            numalign="right",
            floatfmt=".2f",
            intfmt=",",
        )
    )

    # Print summary
    total_bits = sum(s["bits"] for s in stats.values())
    total_numel = sum(s["numel"] for s in stats.values())
    avg_bpw = round(total_bits / total_numel, 2)

    print()
    print(f" -- Average bitrate: {avg_bpw:.2f} bpw")
    print(f" -- Size: {total_bytes / 1024**2:,.2f} MiB")
    print()


def main(args):

    config = Config.from_directory(args.in_dir)
    stc = SafetensorsCollection(args.in_dir)

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
                print(f" -- Overriding from: {o_dir}:")
                for o_key in o_keys:
                    print(f"      {o_key}")
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))
            config.stc = vstc

    # Iterate over all model components (text, vision, etc.)
    all_stats = {}

    for component in config.model_classes:
        model = Model.from_config(config, component=component)

        # Aggregate detailed stats
        component_stats = model.get_detailed_weights_info()
        for layer_name, stat in component_stats.items():
            if layer_name not in all_stats:
                all_stats[layer_name] = {
                    "count": 0,
                    "size_bytes": 0,
                    "numel": 0,
                    "bits": 0.0,
                }
            all_stats[layer_name]["count"] += stat["count"]
            all_stats[layer_name]["size_bytes"] += stat["size_bytes"]
            all_stats[layer_name]["numel"] += stat["numel"]
            all_stats[layer_name]["bits"] += stat["bits"]

    print_markdown_table(all_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir", type = str, default = None, help = "Input model directory")
    parser.add_argument("-or", "--override", type = str, help = "Tensor override spec (YAML)", default = None)
    _args = parser.parse_args()
    main(_args)
