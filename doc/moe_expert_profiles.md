# MoE expert placement profiles

CPU MoE offload (`-mcs` / `-mcl`) splits each block-sparse layer's experts between VRAM and
system RAM. Routing is heavily skewed, so *which* experts stay resident decides how much
traffic crosses the memory bus — on a DDR5-bound setup that is the difference between the CPU
being invisible and the CPU being the bottleneck.

Upstream discovers that ordering at runtime. A **profile** measures it ahead of time, so
placement is right at token zero.

## Why placement matters

Measured on GLM-5.3-Flash (42 MoE layers × 288 experts), sorting each layer's experts by
measured decode traffic:

| experts kept in VRAM | `-mcs` | traffic served from VRAM | CPU share | vs uniform placement |
| --- | --- | --- | --- | --- |
| 75% | 72 | 98.4% | 1.6% | 15.2x less CPU work |
| 50% | 144 | 90.7% | 9.3% | 5.4x |
| 37.5% | 180 | 83.7% | 16.3% | 3.8x |
| 25% | 216 | 73.5% | 26.5% | 2.8x |

The curve is strongly non-linear: half the experts can be evicted while ~91% of the routing
still lands in VRAM.

## Measured effect

Placement only buys throughput where the CPU side is actually the bottleneck. On GLM-5.3-Flash,
`eval/perf.py`, decode tok/s averaged over context 0..4k, native profile, two repeats:

GLM-5.3-Flash 4.05bpw, `-mcs 180`, native profile, decode tok/s:

| context | dynamic | `seed` + profile | |
| --- | --- | --- | --- |
| 0 | 18.80 | 21.88 | +16.4% |
| 256 | 18.52 | 20.79 | +12.3% |
| 512 | 18.82 | 21.00 | +11.6% |
| 1,024 | 18.52 | 19.82 | +7.0% |
| 2,048 | 15.37 | 17.51 | +13.9% |
| 4,096 | 17.19 | 16.76 | -2.5% |
| 8,192 | 18.29 | 20.02 | +9.5% |
| 16,384 | 17.00 | 18.72 | +10.1% |
| 32,768 | 17.05 | 17.95 | +5.3% |
| 65,536 | 15.26 | 14.73 | -3.5% |
| 131,072 | 18.09 | 18.91 | +4.5% |
| 261,888 | 17.40 | 17.69 | +1.7% |

| band | dynamic | `seed` | | |
| --- | --- | --- | --- | --- |
| short, 0-2k | 18.01 | 20.20 | **+12.2%** | wins 5/5 |
| mid, 4k-32k | 17.38 | 18.36 | +5.6% | wins 3/4 |
| long, 64k-256k | 16.92 | 17.11 | +1.1% | wins 2/3 |
| overall | 17.44 | 18.73 | +7.4% | wins 10/12 |

**The gain decays with context, and that is structural.** Placement only affects the MoE share
of a decode step. That work is constant per token while attention grows with context, so by
64k the experts are a small part of the critical path and reordering them buys little. Build
profiles for chat, agentic loops and code completion; skip them for long-document work.

The same run on 2.05bpw `-mcs 150` gives +3.5% (5/7 points, not significant): that config has
DDR5 bandwidth to spare, so there is nothing to reclaim. The 4.05bpw config decodes at ~95% of
its bandwidth ceiling, which is why it responds. Measure your own configuration.

Correctness is unaffected: perplexity over the same rows is 3.489850 with no offload, 3.490881
with offload and dynamic placement, 3.492384 seeded and 3.490442 static -- all within eval
noise.

## Building a profile

```
python util/moe_profile_build.py -m /models/GLM-5.3-Flash-exl3-4.05bpw \
    -corpus /data/wikitext-2-raw/wiki.train.raw -o wiki.npz \
    -nprompts 48 -plen 1024 -gen 128 -resident 108
```

`-corpus` takes a bundled calibration corpus (`wiki`, `c4`, `code`, `multilingual`,
`technical`, `tiny`), a text file, or a JSON list of prompts. The bundled `.utf8` files are not
installed as package data, so on an installed exllamav3 pass a path or set `EXL3_CAL_DATA_DIR`.
A model that does not fit resident needs `-mcs`; routing is unaffected by where experts live:
only routing decisions are needed and a fully resident load is faster.

Decode is the allocation signal — that is the regime offload runs in. Profile quality is driven
first by **how many independent windows** are sampled (`-nprompts`) and only then by how many
tokens each contributes (`-plen`, `-gen`). Diversity beats length: one long generation follows a
single trajectory and its ranking encodes that trajectory rather than the corpus. A profile fitted
that way scored 94.8% against its own counts and delivered 42-45% live — worse than random
placement plus upstream's own sweeps.

Two guards make that failure visible instead of silent:

* **Sample size.** Below 20 hits/expert the tool warns and falls back to the prefill bank, because
  a ranking built from one or two hits per expert is noise that *looks* like extreme concentration
  (unsampled experts sort to the tail, so the "hot" head trivially covers everything).
* **Generalization.** Windows are split into disjoint fit/test sets; capture is reported on the
  held-out half beside `uniform` (random placement) and `oracle` (the best any static profile can
  do). Held-out within 3 points of uniform, or in-sample more than 25 points above held-out, each
  print an explicit warning. With fewer than 4 windows no split is possible and the tool says so
  rather than reporting an in-sample number.

## Flags

| flag | meaning |
| --- | --- |
| `-mcp` / `--moe_cpu_profile` | profile spec: names or paths, comma-separated, optional `:weight` |
| `-mcpm` / `--moe_cpu_profile_mode` | `seed` (default) or `static` |
| `-mcpd` / `--moe_cpu_profile_dir` | extra search directory |
| `-mcpq` / `--moe_cpu_profile_any_quant` | accept a profile from another checkpoint of the same model |

## Using a profile

```
python -m exllamav3.server -m /models/... -mcs 180 \
    --moe_cpu_profile code --moe_cpu_profile_mode seed
```

Profiles are looked up as `<name>.{npz,safetensors,exl3moe,json}`, in order:

1. `<model_dir>/moe_profiles/` — ships inside the model repo
2. `$EXL3_MOE_PROFILE_DIR` (os.pathsep-separated)
3. `~/.cache/exllamav3/moe_profiles/`

### Combining profiles

```
--moe_cpu_profile code:3,wiki:1
```

Sources are merged on **per-layer normalized** frequency, so a corpus with more tokens does not
simply win; the optional `:weight` sets relative contribution.

### seed vs static

| mode | placement | dynamic swapping | use when |
| --- | --- | --- | --- |
| `-mcpm seed` (default) | starts from the profile | stays on, keeps adapting | the workload may drift from the profile |
| `-mcpm static` | frozen at the profile order | off | you want predictable latency and no mid-generation placement changes |

`static` is the more stable of the two: a placement change is a small numeric step for every
migrated expert (the same weights compute slightly differently on the two devices), so `seed`
can shift output subtly at a sweep boundary while `static` never does.

## Profile identity: per model, per quantization

Routing depends on the model **and** on how it was quantized — different bitrates give
different hidden states and therefore different expert choices. A profile therefore belongs to
one `(model, checkpoint)` pair, and both are checked at load:

* **Model identity** — architecture, layer count, expert count, `moe_intermediate_size`,
  `hidden_size`. A mismatch is always fatal; placement from another architecture is meaningless.
* **Checkpoint identity** — `checkpoint_sha` plus `quant_method`/`bits`/`head_bits`/`codebook`.
  A mismatch is fatal unless `--moe_cpu_profile_any_quant` is passed, which downgrades
  it to a loud warning.

`checkpoint_sha` hashes each shard's safetensors **header** (tensor names, dtypes, shapes,
offsets) plus its file size — never the weight data. On a 165 GB, 20-shard checkpoint that is
19.7 MB of reads and about 0.03 s, and it still changes when a checkpoint is requantized at the
same nominal bitrate, which comparing `bits` alone would miss.

A profile without a fingerprint (for example a third-party usage census) loads with a warning
rather than an error.

## Swap policy

With `seed`, dynamic placement continues to run. Its cadence is tunable:

| flag | env | default | meaning |
| --- | --- | --- | --- |
| `-mcw` / `--moe_cpu_swap {on,off}` | `EXL3_MOE_CPU_SWAP` | on | dynamic placement at all |
| `-mcwi` / `--moe_cpu_swap_interval` | `EXL3_MOE_CPU_SWAP_INTERVAL` | 128 | decode steps between sweeps |
| `-mcwm` / `--moe_cpu_swap_max` | `EXL3_MOE_CPU_SWAP_MAX` | 64 | expert swaps per sweep, across **all** layers |
| `-mcwh` / `--moe_cpu_swap_hysteresis` | `EXL3_MOE_CPU_SWAP_HYST` | 2.0 | hit-ratio a CPU expert must beat to be promoted |

Sweeps are deferred to the generator's queue-drained hook so they never fire mid-generation,
with an inline fallback when badly overdue (raw `model.forward` drivers have no such hook).
Note the default budget is 64 swaps *in total* per sweep: on a 42-layer model that is ~1.5
swaps per layer, so converging a cold placement takes thousands of decode steps — which is the
reason to seed from a profile rather than wait.

## Third-party profiles

Profiles are data, not code, and the format is deliberately the same as the usage
census (`counts_decode` / `counts_prefill`, `int64 [prompts, layers, experts]`), so censuses
captured by external tooling load directly. Publish them next to the weights in
`moe_profiles/`, or in a separate repo pointed at by `EXL3_MOE_PROFILE_DIR`.

Positional censuses carry no layer names, so rows are matched to MoE layers in registration
order. Where possible ship the `<name>.meta.json` sidecar our builder writes: it carries
`layer_keys` and the fingerprint, and keyed matching removes any chance of an off-by-one when
an architecture interleaves dense and sparse MLPs.
