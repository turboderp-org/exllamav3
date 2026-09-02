# Optimized quantization recipes

An experimental pipeline for building **per-tensor bitrate recipes** and **in-domain calibration
data**, replacing the uniform/ends-first allocation that `convert.py --bits` uses by default. 
Instead of giving every tensor the same bitrate (with promotions at the head and tail of the
stack to absorb a fractional budget), the pipeline measures how much each individual tensor's
quantization error contributes to the model's end-to-end output error, then solves for the
bit allocation that minimizes the total at a given size budget.

Sensitivities are measured on the unquantized model alone, by injecting noise empirically shown
to mimic the quantizer's error distribution. A full run (trace generation, measurement, 
optimization, conversion) for a ~27B dense model takes roughly two hours on a single fast GPU, 
most of it the conversion itself.

## Important!

**The optimization pipeline is experimental and still a work in progress. It is currently untested
  on sparse models.** 

## Rationale

Three observations drive the design:

1. **Per-tensor quantization sensitivity varies.** The default ends-first promotion order is a 
   reasonable prior for corpus text, but the actual profile depends on the model *and on the data
   distribution*: measured on a model's own chat/reasoning output, the familiar U-shape 
   (first/last layers hottest) largely flattens, and small *mid-stack* attention projections (k/v)
   become the most valuable bits in the network.

2. **Calibration and evaluation data should match deployment.** For heavily-aligned reasoning
   models, raw web corpora are far outside the distribution the deployed model actually
   processes (its own templated prompts and sampled responses). Sensitivities measured on
   raw text transfer poorly, even if formatted by the model's expected chat template. Sensitivities
   measured on a self-sampled trace generalize much better, including to domains absent from the 
   trace.

3. **The whole loop should agree on one objective.** The quantizer's LDLQ error shaping, the
   sensitivity measurement, the recipe optimizer and the evaluation should all target the same
   data distribution. The pipeline uses one self-generated trace for calibration, measurement
   and optimization, and a *disjoint* self-generated trace (new prompts) for evaluation, so 
   overfitting is measured rather than hidden.

## Pipeline overview

```
sc_trace.py                self-sampled trace  ➡  cal_trace.json (qbench-compatible)
                                               ➡  cal_trace.safetensors (packed token rows)
    ⬇

sc_rfn_probe.py            per-tensor quantization error of an existing quant (anchors the noise 
                           amplitudes; optional but recommended)
    ⬇
    
sc_measure.py              per-tensor sensitivity S_t: LDLQ-shaped noise injection into the
                           UNQUANTIZED model, suffix forward, KLD vs clean reference
    ⬇
    
sc_optimize.py             kld(t,K) = S_t * rfn_t(K)^alpha  ➡  exact greedy allocation
                                                            ➡  recipe.yaml
    ⬇
    
convert.py                 -rcp recipe.yaml -cd cal_trace.safetensors

    ⬇
    
eval/qbench_prompts.py     held-out eval trace (new prompts + optional tool calls)
eval/qbench.py             KLD/ppl vs the bf16 reference, per-quant comparison
```

Every stage writes plain files, so stages can be rerun or swapped independently. The
measurement and rfn anchors are properties of the model (not of any one bitrate target), so one
measurement serves recipes at every target bitrate.

## Walkthrough

The examples assume a model at `models/foo/hf` with existing EXL3 conversions, optionally, under
`models/foo/exl3/`, and a Hessian-friendly GPU (the measurement holds the unquantized model
plus per-layer Hessians; a 27B model in bf16 fits comfortably in 96 GB).

### 1. Generate the trace

```
python sd_trace.py \
    -m models/foo/exl3/6.00bpw \
    -o models/foo/exl3/cal_trace.json \
    -co models/foo/exl3/cal_trace.safetensors \
    -cs 262144 \
    -ambs 32
```

Samples the model on ~140 seeded conversations spanning subjects, task formats and 19
languages, using the model's own chat template and default sampling, batched through the
dynamic generator. Use the unquantized model or a high-bitrate quant (6 bpw is minimally
distorted and much faster). Produces:

- `cal_trace.json` — variable-length (context, response) rows, directly usable as a qbench
  `test_trace`.
- `cal_trace.safetensors` — the exact model-visible token streams (templated context + sampled
  response, reasoning included), shuffled and packed into fixed calibration rows
  (default 250 x 2048).

Useful arguments:

- **-cr / --cal_rows**, **-cc / --cal_cols**: packed row count/width, default 250 x 2048
  (matching `convert.py` defaults).
- **--max_new_tokens**: per-turn cap, default 3072.
- **--max_epochs**: if the seed set doesn't reach the token target, additional epochs resample
  the same conversations with different seeds. Default 4.
- **-tv / --template_vars**: chat template variables as JSON, merged over
  `{"enable_thinking": true, "reasoning_effort": "high"}`. Variables a template rejects are
  dropped automatically with a warning (e.g. templates that only accept specific
  `reasoning_effort` levels).
- **-ambs**: for recurrent/hybrid models this sets the number of recurrent state slots and
  therefore generation concurrency; 32 is a good default for a fast GPU.

See also `python sd_trace.py -h` for more inference options, including draft model settings.

### 2. Probe the quantizer's error amplitudes (recommended)

```
python sc_rfn_probe.py \
    -mq models/foo/exl3/2.00bpw \
    -mr models/foo/hf \
    -d 0 \
    -o models/foo/exl3/rfn_2.00bpw.json
```

Dequantizes every tensor of an existing conversion and records its actual relative weight
error `rfn = ||W_q - W|| / ||W||`. This anchors the noise model per tensor: how hard each
tensor actually is to quantize (typically a ±10% spread at fixed K, with structure by module
type). 

Any existing conversion works; K=2 anchors have been validated to extrapolate across
the full K range via the amplitude-halving law (see "Model" below). Without a probe, the
optimizer falls back to a global anchor (`--anchor`, default `2:0.292`).

### 3. Measure per-tensor sensitivity

```
python sc_measure.py \
    -m models/foo/hf \
    -d 0 \
    --shaped \
    -tr models/foo/exl3/cal_trace.safetensors \
    -rr models/foo/exl3/rfn_2.00bpw.json \
    -rs "1.0,0.5" \
    -o models/foo/exl3/noise_attrib.json
```

For every quantizable Linear, perturbs the weights in place with seeded noise at the anchored
amplitude, runs the rest of the model from a cached boundary state, and records the KLD of the
final logits against the clean reference. One suffix pass per tensor per noise level; a 27B
model with two levels takes ~35 minutes on a fast GPU. Output is written incrementally and the
script resumes from a partial file.

- **--shaped**: strongly recommended. Captures the same per-layer Hessians the quantizer uses
  (identical damping, sign flips, block Hadamard and block-LDL) over `-hr/--h_rows` extra
  trace rows, and samples noise with the LDLQ error distribution `dW = P^T L^-T eta` plus
  per-output-channel scaling. Without it, isotropic noise overestimates sensitivity by up to
  ~6x on tensors with strongly anisotropic input statistics (v_proj tends to be worst), because
  real LDLQ error is steered away from directions the data exercises. Shaped-noise predictions
  validate at Spearman 0.84 against ground-truth swap attribution of a real conversion
  (0.61 unshaped), with the residual error nearly uniform across tensors (harmless to the
  allocation).
- **-tr / --trace**: use the packed trace for both the evaluation rows and the Hessian capture
  rows. Omit to fall back to wikitext2 — but see "Rationale"; wikitext-measured recipes
  transfer poorly.
- **-rs / --rfn_scale**: injection amplitudes as multiples of each tensor's anchor. Two levels
  an octave apart (default `1.0,0.5`) give a per-tensor scaling-exponent sanity check.

### 4. Compile a recipe

```
python sc_optimize.py \
    -m models/foo/exl3/noise_attrib.json \
    -rr models/foo/exl3/rfn_2.00bpw.json \
    -b 3.14 \ 
    -hb 4 \
    -al 2.0 \
    -mink 2 \
    -o models/foo/exl3/recipe_2.50bpw.yaml
```

Models each tensor's KLD contribution at every bitrate as `kld(t, K) = S_t * rfn_t(K)^alpha`
and solves the size-budget allocation exactly (greedy by marginal KLD reduction per stored
bit, which is optimal since the curves are convex, plus an exchange-repair pass for
budget-boundary granularity). Prints the predicted KLD against an ends-first baseline at the
same budget, and writes a YAML recipe mapping every quantizable tensor to an integer bitrate.

Key arguments:

- **-b / --bitrate**: target mean bits per weight over the budgeted (non-head) tensors, same
  accounting as `convert.py --bits`.
- **-hb / --head_bits**: head bitrate (integer), recorded in the recipe (default 6).
- **-mink / --min_k**: minimum bitrate for individual tensors. Should not be needed but it can
  be useful for some models to set `min_k = floor(bitrate)` if optimization ends up 
  underestimating the cost of demoting tensors.
- **-al / --alpha**: KLD-vs-noise scaling exponent. The theoretical value is 2 (quadratic);
  fitted values from two-level measurements run slightly low (~1.7–1.9) and real-quant
  transitions measure ~2.0, so `--alpha 2.0` is recommended over the default fit.
- **-all / --alpha_low**: exponent for the curve *below* the anchor K (i.e. pricing demotions
  into high-noise territory). Measured ~2.35 for the K2->K1 octave on Qwen3.8-27B. The law 
  steepens once a tensor is mostly noise. Only relevant with `--min_k` below the anchor.
- **-tie / --tie**: suffix groups forced to share a bitrate because the fast inference paths
  fuse their GEMMs. Default `k_proj+v_proj,gate_proj+up_proj`, tied within the same parent
  module; KLD cost tends to be tiny. Pass `""` to optimize every tensor independently, forgoing
  some performance for the resulting quant.
- **-t / --table**: optionally dump the full predicted (tensor x K) KLD table as JSON, for
  external solvers or analysis.

The recipe file is plain YAML:

```yaml
target_bpw: 3.14
achieved_bpw: 3.1398
head_bits: 4
tensors:
  model.layers.0.self_attn.q_proj: 2
  model.layers.0.self_attn.k_proj: 3
  model.layers.0.self_attn.v_proj: 3
  ...
```

### 5. Convert with the recipe and the trace

```
python convert.py \
    -i models/foo/hf -w /tmp/work \
    -o models/foo/exl3/2.50bpw_opt \
    -rcp models/foo/exl3/recipe_2.50bpw.yaml \
    -cd models/foo/exl3/cal_trace.safetensors
```

- **-rcp / --recipe**: use the recipe in place of `--bits`. The recipe must cover the model's
  budgeted tensors exactly (missing or unknown keys abort, since either means the recipe was
  built for a different model). `--head_bits` defaults to the recipe's value; an explicit flag
  wins. The tensor map is stored in the job state, so resumed jobs are immune to recipe file
  edits, and `--hq`/`--bits` are ignored with a warning.
- **-cd / --cal_data**: calibrate on the packed trace instead of the bundled corpus mix. This
  is where the LDLQ compensation gets pointed at the deployment distribution.

Both flags work independently: `-cd` alone converts a uniform-bitrate model with in-domain
calibration; `-rcp` alone applies a recipe with the default calibration mix.

See also: [convert.md]

### 6. Evaluate on held-out data

Because the quant was optimized *and* calibrated on `cal_trace`, evaluating on that same trace
would overstate the gain. `eval/qbench_prompts.py` generates a disjoint trace — an entirely
different prompt set over the same distribution, with an optional tool-calling portion
(`--tool_frac`, default 0.3: conversations that define OpenAI-style tools, elicit a reasoning
trace plus a tool invocation, and feed scripted tool-role results back in).

Similarly to `sd_trace.py`, a faithful, high precision quant serves well for producing the 
in-domain evaluation trace,

```
python eval/qbench_prompts.py \
    -m models/foo/exl3/6.00bpw \
    -o models/foo/qbench_prompts_gen.json \
    --min_tokens 45000 \
    --max_new_tokens 2560
```

Point a qbench project's `test_trace` at the file and compare the optimized quant against the
uniform conversions (see `eval/qbench_example.yaml`). KLD against the unquantized reference,
over the sampled response positions only, is the primary metric; per-token KLD vectors are
cached so slices (e.g. tool rows vs prose rows) can be analyzed afterwards.
