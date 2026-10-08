"""
Shared helpers for the exllamav3 test suite. Tests import from here instead of carrying their own copies of
device selection, model paths, tolerance helpers and reference implementations. A helper moves here as soon as a
second test file needs it.

Infrastructure
    env           hardware/software queries behind the skip markers; get_test_device() for module-level code
    models        the canonical test model registry (tests/models.yaml + local overrides)
    compare       tolerance helpers and logit metrics (rel_err, assert_close_mr, kl_divergence, top1_agreement)
    checkpoint    synthetic tensors served through the real SafetensorsCollection (module_config)
    isolated      run a test function in a fresh interpreter with its own environment (run_isolated)
    repo_scripts  import scripts from util/ and eval/ by path

Synthetic weights and references, by subsystem
    exl3          random EXL3 linears/experts, dequantization, dense reference, MLP activations
    attention     fp32 attention reference, random paged / packed-quantized caches, quantized round trip
    moe           expert pointer tables, routing layouts, swizzle, per-expert MLP reference
    moe_cpu       CPU expert-kernel ISA tiers (run_per_tier), swizzled layouts, thread/affinity fixture
    routing       router configs and tie-aware selection comparison
    mla           random MLAttention modules, explicit per-head reference forward, paged drivers
    dsv4          tiny DeepSeek-V4 model helpers (module loop, cached forward, noise floor)
    tiny_models   tiny random checkpoints in native formats (DeepSeek-V4)
    cache         fake cache layers and pages for page-table / CPU-tier tests
    tp            fakes for tensor-parallel export/import tests

Model level
    e2e           load_model through model_init (configurations as CLI arguments), teacher-forced logits,
                  greedy runs, near-tie-tolerant comparison, per-model noise floor
    graph         lower-level loading with several caches per model, decode-vs-forward check for graph paths
"""
