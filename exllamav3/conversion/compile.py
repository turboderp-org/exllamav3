import os
import shutil
import json
from ..loader.safetensors import SafetensorsCollection
from ..version import __version__
from safetensors.torch import save_file
from ..util.memory import free_mem
from ..modules import Module
from .quant_config import update_config, create_quantization_config_json

def tsize(t):
    return t.nelement() * t.element_size()

def dsize(d):
    size = 0
    for _, v in d.items(): size += tsize(v)
    return size

def compile_model(args, model, config, tokenizer):

    in_dir = args["in_dir"]
    out_dir = args["out_dir"]
    if args.get("model_stc"):
        qtensors_stc = config.stc
    else:
        work_dir = args["work_dir"]
        qtensors_dir = os.path.join(work_dir, "qtensors")
        qtensors_stc = SafetensorsCollection(qtensors_dir)

    # Prepare output directory
    if not os.path.exists(out_dir):
        print(f" -- Creating directory {out_dir}")
        os.makedirs(out_dir)
    else:
        print(f" -- Writing into {out_dir}")
        if len(os.listdir(out_dir)) != 0:
            print(f" !! Warning, output directory is not empty")

    # Allocate shards
    total_size = 0
    max_shard_bytes = args["shard_size"] * 1024**2
    out_map = []
    out_map.append([])
    current_shard_size = 0
    for module in model.modules:
        prefix = module.key
        sizes = qtensors_stc.get_tensor_sizes(prefix)
        if len(sizes) == 0:
            continue
        size = sum(sizes)
        if size > max_shard_bytes:
            print(f" !! Warning, unable to fit module {module.key} in single shard of {args['shard_size']} MB")
        if current_shard_size + size > max_shard_bytes and current_shard_size > 0:
            current_shard_size = 0
            out_map.append([])
        current_shard_size += size
        total_size += size
        out_map[-1].append(module)

    # Additional tensors
    extra_tensors = {}
    for _, cls in config.model_classes.items():
        extra_tensors.update(cls.get_additional_compiled_tensors(config))
    for key, data in extra_tensors.items():
        size = data["n_bytes"]
        if size > max_shard_bytes:
            print(f" !! Warning, unable to fit module {module.key} in single shard of {args['shard_size']} MB")
        if current_shard_size + size > max_shard_bytes and current_shard_size > 0:
            current_shard_size = 0
            out_map.append([])
        current_shard_size += size
        total_size += size
        out_map[-1].append(key)

    # Write model tensors
    map_dict = {}
    num_files = len(out_map)
    for file_idx, modules in enumerate(out_map):
        filename = (
            "model.safetensors" if num_files == 1 else
            f"model-{file_idx+1:05}-of-{num_files:05}.safetensors"
        )
        print(f" -- Writing {filename}")
        file_dict = {}
        for m in modules:
            if isinstance(m, Module):
                prefix = m.key
                tensors = qtensors_stc.get_tensors(prefix, allow_bf16 = True)
                tensors = {k: v.contiguous() for k, v in tensors.items()}
                qtensors_stc.close()
            elif isinstance(m, str):
                tensor = config.stc.get_tensor(m, allow_bf16 = True)
                tensors = {m: tensor.contiguous()}
            file_dict.update(tensors)
        for name in file_dict.keys():
            map_dict[name] = filename
        save_file(file_dict, os.path.join(out_dir, filename))
        del file_dict
        free_mem()

    # Copy non-tensor files
    print(f" -- Copying non-tensor files from {in_dir}")
    filtered_files = []
    ignored_files = []
    for f in os.listdir(in_dir):
        if not os.path.isfile(os.path.join(in_dir, f)):
            continue
        if f.endswith(".safetensors"):
            continue
        if f == "config.json":
            continue
        if f == "model.safetensors.index.json":
            continue
        if any(f.endswith(x) for x in [".bin", ".ckpt", ".pth", ".pt"]):
            ignored_files.append(f)
            continue
        filtered_files.append(f)
    for f in filtered_files:
        print(f"     - {f}")
        source_file_path = os.path.join(in_dir, f)
        target_file_path = os.path.join(out_dir, f)
        shutil.copy(source_file_path, target_file_path)
    if ignored_files:
        print(f" !! Warning, the following file(s) will not be included in output model:")
        for f in ignored_files[:10]:
            print(f"     - {f}")
        if len(ignored_files) > 10:
            print(f"     - (+ {len(ignored_files) - 10} more)")

    # Write new model.safetensors.index.json maybe
    if num_files > 1:
        print(f" -- Writing model.safetensors.index.json")
        safetensors_index = {
            "metadata": {
                "total_size": total_size,
            },
            "weight_map": map_dict
        }
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(safetensors_index, indent = 4))

    # Update and write config.json
    print(f" -- Writing config.json")
    with open(os.path.join(in_dir, "config.json"), "r") as f:
        config_dict = json.load(f)

    qcfg = {
        "quant_method": "exl3",
        "version": __version__,
        "bits": args["bits"],
        "head_bits": args["head_bits"],
    }
    if "cal_rows" in args:
        qcfg.update({
            "calibration": {
                "rows": args["cal_rows"],
                "cols": args["cal_cols"],
            }
        })
    if "apply_out_scales" in args:
        qcfg.update({
            "out_scales": {True: "always", False: "never", None: "auto"}[args["apply_out_scales"]]
        })
    if "codebook" in args:
        qcfg.update({
            "codebook": args["codebook"]
        })

    if "quantization_config" in config_dict:
        orig_qcfg = config_dict["quantization_config"].copy()
        if orig_qcfg.get("quant_method") == "exl3":
            qcfg = orig_qcfg
            qcfg.update({
                "bits": args["bits"],
                "head_bits": args["head_bits"],
            })
            if "codebook" in args:
                qcfg.update({
                    "codebook": args["codebook"]
                })
        else:
            qcfg.update({
                "original_quantization_config": orig_qcfg
            })

    update_config(config_dict)
    config_dict["quantization_config"] = qcfg
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(json.dumps(config_dict, indent = 4))

    # Add extra metadata to quant_config
    print(f" -- Creating quantization_config.json")
    create_quantization_config_json(out_dir)

    print(f" -- Finished compiling model to {out_dir}")
