# Residual control-vector projection

Qwen4Exp (`Qwen3.8-Flash-Next`) can apply an optional per-layer projection to
each FP32 hyperconnection stream after a decoder block. This changes inference
behavior while preserving the model's stored weights. A vector trained to remove
a refusal direction is one possible use; neither answer quality nor a reduction
in refusals follows from implementing the projection correctly.

The first integration supports the Qwen4Exp text component. Tensor parallelism,
layer remapping and quantization capture are unsupported with projection enabled.
There is no per-request toggle. Draft-model integration and other architectures
are outside the tested scope.

## Python API

```python
from exllamav3 import Config, ControlVector, Model

config = Config.from_directory("/path/to/exl3-model")
vector = ControlVector.from_file(
    config,
    "/path/to/vectors/control.safetensors",
    scale=1.0,
    layer_range=(4, 44),
)
model = Model.from_config(config, control_vector=vector)
# Load the model and create its cache/generator as usual.
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

The inference API consumes safetensors, not GGUF. For an F32 `controlvector` GGUF,
copy the `direction.N` arrays into safetensors without changing their values and
add the metadata above. No vector weights are bundled with the library.

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
In-memory KV/recurrent caches belong to the model/generator built for that vector;
do not replace projection tensors on a live model.

## Validation

From a checkout with its matching native extension built:

```sh
pytest tests/modules/control_vector tests/model_setup/test_control_vector.py
```

The tests use generated vector files and a synthetic configuration. They cover
FP64-reference agreement, independent streams, noncontiguous input, invalid files,
CUDA graph replay, layer indexing with PLE, disabled graph parity, ownership of
directions and persistent-cache identity. GPU tests use the suite's `device`
fixture. CPU-only cases are marked `nogpu`.

These tests do not establish generation quality, refusal rates, end-to-end
throughput or parity between the vector's original engine and a quantized EXL3
model. For performance comparisons, use identical token sequences with warm
kernels and report both prefill/decode cost and memory usage; free-running answers
can differ and are not a controlled measure of projection overhead.
