import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.loader.safetensors import SafetensorsCollection
import argparse
import json

def main(args):
    stc = SafetensorsCollection(args.model_dir)
    if len(stc.file_headers) == 0:
        print(" -- No .safetensors files found, skipping")
        return
    if len(stc.file_headers) <= 1 and not args.force:
        print(" -- Only one .safetensors file found, skipping (use --force to override)")
        return

    total_size = 0
    map_dict = {}
    for k, v in stc.tensor_file_map.items():
        basename = os.path.basename(v)
        offsets = stc.file_headers[v][k]["data_offsets"]
        size = offsets[1] - offsets[0]
        total_size += size
        map_dict[k] = basename

    # New dict
    safetensors_index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": map_dict
    }

    # Write new model.safetensors.index.json maybe
    filename = os.path.join(args.model_dir, "model.safetensors.index.json")
    update = os.path.exists(filename)
    with open(filename, "w") as f:
        f.write(json.dumps(safetensors_index, indent = 4))
    if update:
        print(f"Updated {filename}")
    else:
        print(f"Created {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    parser.add_argument("-f", "--force", action = "store_true", help = "Write index even if there is only one .safetensors file in the model directory")
    _args = parser.parse_args()
    main(_args)