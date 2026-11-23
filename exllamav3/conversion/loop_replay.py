"""Serial calibration traversal of a shared-weight Loop region."""
import torch


def replay_region(rows, modules, input_ids, bad_rows, load_module, unload_module, label):
    """Run one visit through a region; callers choose original or quantized weights.

    Rows are owned by the caller and replaced as each layer advances. No recurrent
    inference cache is used: each calibration row starts at position zero.
    """
    for module in modules:
        print(f' -- {label}: {module.key}', flush=True)
        load_module(module)
        try:
            for i, row in enumerate(rows):
                if i in bad_rows:
                    continue
                params = {'attn_mode': 'flash_attn_nc', 'input_ids': input_ids[i], 'quant_preserve': {}}
                x = module.prepare_for_device(row, params)
                result = module.forward(x, params)
                if not torch.isfinite(result).all().item():
                    raise ValueError(f'Non-finite Loop calibration state in row {i}, {module.key}')
                rows[i] = result.cpu()
        finally:
            unload_module(module)
    return rows
