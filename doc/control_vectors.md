# Residual control-vector projection

Qwen4Exp (`Qwen3.8-Flash-Next`) can apply an optional per-layer projection to
each FP32 hyperconnection stream after a decoder block. This changes inference
behavior while preserving the model's stored weights. A vector trained to remove
a refusal direction is one possible use; neither answer quality nor a reduction
in refusals follows from implementing the projection correctly.

The first integration supports the Qwen4Exp text component. Tensor parallelism,
layer remapping and quantization capture are unsupported with projection enabled.
There is no per-request toggle. Speculative decoding (draft model, MTP, n-gram or
hybrid drafting) is rejected when projection is enabled. Other architectures are
unsupported.

## Command line

Examples using the shared model loader accept three optional arguments:

```sh
python examples/chat.py -m /path/to/exl3-model -mode qwen35 \
    --control_vector /path/to/vectors/control.safetensors \
    --control_vector_scale 1.0 \
    --control_vector_layer_range 4 44
```

Use the normal memory/offload and cache options for your hardware. Scale defaults
to `1.0`; the range defaults to all directions in the file. A scale or range
without a vector file is an error. Incompatible loading options are rejected
before loading the weights.

## Python API

```python
from exllamav3 import Cache, Config, ControlVector, Generator, Model, Tokenizer

config = Config.from_directory("/path/to/exl3-model")
vector = ControlVector.from_file(
    config,
    "/path/to/vectors/control.safetensors",
    scale=1.0,
    layer_range=(4, 44),
)
model = Model.from_config(config, control_vector=vector)
cache = Cache(model, max_num_tokens=8192, max_batch_size=1)
tokenizer = Tokenizer.from_config(config)
model.load(max_batch_size=1, max_chunk_size=2048)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                      max_batch_size=1, max_chunk_size=2048)
try:
    print(generator.generate(
        prompt="The capital of France is",
        max_new_tokens=32,
        completion_only=True,
    ))
finally:
    generator.close()
    model.unload()
```

The layer range is **zero-based and inclusive**: `(4, 44)` selects 41 decoder
blocks. It counts decoder layers, not positions in `model.modules`; PLE modules
and the projection modules themselves do not change these indices. Omitting the
range selects the entire contiguous range of directions in the file.

Omitting `control_vector`, or loading a valid vector with `scale=0`, constructs
the stock module graph and retains the stock persistent-cache identity. Strength
must be finite and nonnegative. The vector is selected when the model is created;
construct a new model and cache/generator to change it. No environment variables
or server-specific configuration are required by this API.

## Vector format and math

Keep vector files **outside the model checkpoint directory**, where a checkpoint
loader could otherwise interpret them as model weights. The file uses safetensors
with one F32 tensor of shape `[hidden_size]` for each included decoder layer,
named `direction.0`, `direction.1`, etc. Every selected layer must be present.

Required string metadata:

```json
{
  "source_architecture": "qwen4_exp",
  "operation": "project",
  "hidden_size": "2560",
  "num_hidden_layers": "48",
  "hc_count": "4"
}
```

Dimensions must match the loaded model. The example values describe the original
Flash Next; the loader reads actual dimensions from the configuration.

For raw F32 vector `v` and requested strength `s`, the implementation computes
`w = float32(s * v)`, takes its norm in FP64, then stores a normalized F32
direction `u` and the effective norm `a` in F32. Each residual stream `h` becomes:

```text
h <- h - a * dot(h, u) * u
```

This preserves the scaling/normalization semantics of the per-layer projection
variant used by [Cudecnik's Flash Next vector package](https://huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection).
It is projection, not additive steering. A non-unit input vector's magnitude
contributes to effective strength. At unit effective strength this removes the
component parallel to the direction. File metadata checks do not establish that
a particular trained vector is suitable for a particular checkpoint.

## Converting a GGUF vector

The inference API consumes safetensors. The optional converter copies F32
`direction.N` arrays from a `controlvector` GGUF without changing their values,
normalizing them or shifting their layer indices. It obtains the metadata above
from the chosen model's `config.json`:

```sh
python -m pip install gguf
python util/convert_control_vector.py control.gguf control.safetensors \
    --model_dir /path/to/exl3-model
```

Run the conversion outside the checkpoint directory. Existing output files are
never overwritten. The converter rejects wrong dimensions, non-F32 tensors,
nonfinite or zero vectors, unknown tensor names and gaps in the layer indices.
Conversion checks the container and shape; the source must already be a
per-layer projection vector suitable for this checkpoint. It does not convert
additive steering semantics into projection. `gguf` is needed only for this
utility, and no vector weights are bundled with the library.

## Execution and cache behavior

The projection is a separate module with a Triton GPU kernel and a CPU reference.
Its insertion breaks the precomputed weighted-copy handoff between the affected
hyperconnection sites. Pending residual updates are completed and stale weighted
copies invalidated before projection, so the following block sees the modified
state. The last KV-producing module remains the last decoder block even when a
projection follows that block.

Loaded directions are held as immutable bytes, and each model gets its own
tensors. Replacing or moving the source file does not change an existing model.
Persistent-cache fingerprints include the effective directions, strengths,
dimensions and selected layer indices. Stock and projected runs, or different
projections over the same weights, therefore use different disk-cache identities.
Fingerprints are computed from validated data even when constructing a
`ControlVector` directly; they cannot be supplied as a constructor argument.
In-memory KV/recurrent caches belong to the model/generator built for that vector;
do not replace projection tensors on a live model.

## Validation

From a checkout with its matching native extension built:

```sh
pytest tests/modules/control_vector tests/model_setup/test_control_vector.py \
    tests/model_setup/test_control_vector_cli.py \
    tests/conversion/test_control_vector_file.py \
    tests/e2e/test_control_vector_inference.py
```

The tests use generated vector files and a tiny random-weight Qwen4Exp checkpoint
with GDN, QSA, hyperconnections and PLE. No model download is needed. They cover
FP64-reference agreement, independent streams, noncontiguous input, invalid files,
CUDA graph replay, pending residual updates, layer indexing with PLE, disabled
graph/logit parity, ownership of directions, CLI rejection paths and converter
value preservation. Inference tests compare full forward with chunked prefill
and batched decode, reuse a shared prefix, restore it from disk in a fresh model,
and check that a changed vector cannot reuse that cache. Continuations are fixed
when comparing logits. GPU tests use the suite's `device` fixture. CPU-only cases
are marked `nogpu`; conversion tests skip if the optional `gguf` package is absent.

These tests do not establish generation quality, refusal rates, end-to-end
throughput or parity between the vector's original engine and a quantized EXL3
model. For performance comparisons, use identical token sequences with warm
kernels and report both prefill/decode cost and memory usage; free-running answers
can differ and are not a controlled measure of projection overhead.
