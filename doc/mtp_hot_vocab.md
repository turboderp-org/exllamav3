# Qwen MTP hot vocabulary

Qwen3.5/3.6 MTP evaluates the target output projection for each draft step. For models with a
large vocabulary, `MTPHotVocabConfig` can build a smaller EXL3 output head from selected
128-token groups and copy the matching input embeddings to the GPU.

The reduced head proposes greedy draft tokens. The target runs its full output head and verifies
each proposal, as in standard speculative decoding. Vocabulary selection can change acceptance
and speed. Target verification preserves the target distribution.

## Constraints

- Qwen3.5/3.6 with its embedded MTP component
- EXL3 target `lm_head`
- layer-split inference on one GPU
- complete 128-token EXL3 Hadamard groups
- enough VRAM for the reduced head and copied embeddings

The existing MTP path remains the default.

## Build a block map

Use text from the expected workload and, when available, proposal IDs from unrestricted MTP:

```bash
python util/build_mtp_hot_blocks.py \
    -m /models/qwen-exl3 \
    -c /repos/project \
    --draft_ids draft-proposals.txt \
    -b 4096 \
    -o mtp-hot-blocks.txt
```

`-b 4096` selects 65,536 tokens. The script ranks groups by proposal frequency, then corpus
frequency, and retains groups that contain special tokens. Maps depend on the model tokenizer
and workload.

Proposal files contain one token ID per line. Construct `Generator` with
`record_draft_stats = True`, then read `job.draft_token_ids` after generation. The proposal
trace is optional.

## Configuration

```python
from exllamav3 import Model, MTPHotVocabConfig

hot = MTPHotVocabConfig(
    blocks_path = "mtp-hot-blocks.txt",
    embedding_dtype = "fp8",
)
draft_model = Model.from_config(
    config,
    component = "mtp",
    mtp_hot_vocab_config = hot,
)
```

`model_init` exposes `--mtp_hot_blocks`, `--mtp_hot_embedding_dtype` and
`--mtp_hot_validate`. The validation option evaluates both draft heads and removes the speed
benefit.

Servers can use the matching environment variables:

| Variable | Meaning |
| --- | --- |
| `EXL3_MTP_HOT_BLOCKS` | Path to a packed-block map |
| `EXL3_MTP_HOT_EMBED_DTYPE` | `fp16` (default) or `fp8` |
| `EXL3_MTP_VALIDATE_SUBHEAD` | Nonzero enables full-head validation |

## Tuning

FP8 embeddings reduce the copied embedding allocation and can improve draft speed on supported
GPUs. Measure generation rate, acceptance and peak VRAM across the intended prompt mix. A narrow
map can lose on other languages or domains.

The reduced head uses extra VRAM even though each draft step costs less. Long contexts also shift
more time into attention and KV work, so benchmark draft depth at the intended maximum context.
