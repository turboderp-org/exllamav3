"""Copy F32 GGUF control directions into the Qwen4Exp projection format.

Requires the optional gguf package only when converting, not during inference.
The input must contain projection directions suitable for the chosen checkpoint;
shape checks do not establish their behavioral suitability.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from safetensors.numpy import save


def convert(source: Path, destination: Path, model_dir: Path):
    from gguf import GGUFReader, GGMLQuantizationType

    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("Conversion currently supports Qwen4Exp only")
    text = config["text_config"]
    width, layers, streams = text["hidden_size"], text["num_hidden_layers"], text.get("hc_count", 4)
    if any(type(v) is not int or v <= 0 for v in (width, layers, streams)):
        raise ValueError("Invalid model dimensions")
    if destination.resolve().parent == model_dir.resolve():
        raise ValueError("Keep control vectors outside the model checkpoint directory")
    reader = GGUFReader(str(source))
    architecture = reader.fields.get("general.architecture")
    if architecture is None or architecture.contents() != "controlvector":
        raise ValueError("Expected a controlvector GGUF")
    tensors = {}
    for tensor in reader.tensors:
        match = re.fullmatch(r"direction\.(0|[1-9][0-9]*)", tensor.name)
        if match is None or not 0 <= int(match[1]) < layers or tensor.name in tensors:
            raise ValueError(f"Invalid or duplicate direction: {tensor.name}")
        if tensor.tensor_type != GGMLQuantizationType.F32 or tuple(tensor.shape) != (width,):
            raise ValueError(f"Expected an F32 vector of width {width}: {tensor.name}")
        value = np.array(tensor.data, dtype = np.float32, copy = True).reshape(width)
        if not np.isfinite(value).all() or np.linalg.norm(value.astype(np.float64)) == 0:
            raise ValueError(f"Invalid direction values: {tensor.name}")
        tensors[tensor.name] = value
    if not tensors:
        raise ValueError("Control vector contains no directions")
    indices = sorted(int(k.split(".")[1]) for k in tensors)
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Control-vector directions must be contiguous")
    data = save(tensors, metadata = {
        "source_architecture": "qwen4_exp", "operation": "project",
        "hidden_size": str(width), "num_hidden_layers": str(layers), "hc_count": str(streams),
    })
    # Exclusive creation also prevents overwriting the input or an existing vector.
    with destination.open("xb") as stream:
        stream.write(data)
    return len(tensors)


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("source", type = Path)
    parser.add_argument("destination", type = Path)
    parser.add_argument("--model_dir", type = Path, required = True)
    args = parser.parse_args()
    try:
        count = convert(args.source, args.destination, args.model_dir)
    except ImportError:
        parser.error("Conversion requires the optional gguf package: python -m pip install gguf")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Copied {count} F32 directions to {args.destination}")


if __name__ == "__main__":
    main()
