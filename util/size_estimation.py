import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model
import argparse
from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection
import yaml


def tsize(t):
    return t.nelement() * t.element_size()


def dsize(d):
    size = 0
    for _, v in d.items(): size += tsize(v)
    return size


def main(args):

    # Config/model
    config = Config.from_directory(args.in_dir)
    model = Model.from_config(config)

    # Tensor collection
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

    # New bpw etc.
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()
    bpw_layer = round(bpw_layer, 2)
    bpw_head = round(bpw_head)
    print(f" -- New estimated model bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")
    print(f" -- VRAM: {vram_bits / 8 / 1024**3:.0f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir", type = str, default = None, help = "Input model directory")
    parser.add_argument("-or", "--override", type = str, help = "Tensor override spec (YAML)", default = None)
    _args = parser.parse_args()
    main(_args)
