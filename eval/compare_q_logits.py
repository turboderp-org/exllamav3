import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3 import Config, Model, Cache, Tokenizer, model_init
import torch
import torch.nn.functional as F
import math
import json

from compare_q import get_test_data, save_tensor

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_purple = "\u001b[35;1m"
col_cyan = "\u001b[36;1m"
col_white = "\u001b[37;1m"

def stream_forward(args, config, model, batch):

    state = batch
    for idx, module in enumerate(model.modules):

        # Load next module
        print(f" -- Loading module: {col_green}{module.key}{col_default}")
        config.stc.begin_deferred_load()
        module.load(torch.device(args.device) if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()

        # Forward pass
        print(f" -- Forward pass")
        params = {}
        state = module.prepare_for_device(state, params)
        state = module.forward(state, params)

        # Unload current module
        module.unload()
        config.stc.close()

    return state

@torch.inference_mode()
def main(args):

    # Create model config
    config = Config.from_directory(args.model_dir)
    config.override_dynamic_seq_len(2048)
    # tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)

    # Input state
    with open(args.dataspec, "r", encoding = "utf8") as f:
        data_spec = json.load(f)
    eval_ids = get_test_data(data_spec)
    eval_ids = eval_ids[:args.rows]

    collect_logits = []
    batches = eval_ids.split(args.rows_per_batch, 0)
    for idx, batch in enumerate(batches):
        print(f" -- Forward pass {idx + 1} / {len(batches)}")
        logits = stream_forward(args, config, model, batch)
        collect_logits.append(logits.cpu())
        del logits

    collect_logits = torch.cat(collect_logits, dim = 0)
    collect_logits = collect_logits.split(1, 0)

    print(f" -- Writing {args.out_logits}")
    save_tensor(collect_logits, args.out_logits)
    print(f" -- Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    parser.add_argument("-d", "--dataspec", type = str, help = "Data specification (JSON file)")
    parser.add_argument("-dev", "--device", type = int, help = "CUDA device index", default = 0)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 10)
    parser.add_argument("-rpb", "--rows_per_batch", type = int, help = "Rows per batch", default = 5)
    parser.add_argument("-o", "--out_logits", type = str, help = "Output file", required = True)
    _args = parser.parse_args()
    main(_args)
