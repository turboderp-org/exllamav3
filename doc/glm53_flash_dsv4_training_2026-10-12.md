# GLM-5.3-Flash and DeepSeek-V4-Flash training support (2026-10-12)

Branch `feat/glm53-flash-dsv4` (on top of `feat/glm4-moe`). Asked for: "GLM 5.3
and DS v4 flash". "GLM 5.3" proper (zai-org/GLM-5.3, 500B `GlmMoeDsaForCausalLM`)
cannot run on this box at any quant (>= 125 GB at 2 bpw vs 125 GB RAM), so the
target is GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, 320B, 45 layers).
DeepSeek-V4-Flash is `DeepseekV4ForCausalLM` (284B, 43 layers); the only exl3
quant of that trunk is turboderp's Vision-Exp variant (text trunk + a vision
tower the trainer never loads). DeepSeek-V4.1-Flash is a different 763B arch
and is out of scope.

## What was missing (and now exists)

Both archs were rejected by `assert_block_supported` for the same reasons.
Everything below is a plain-torch, autograd-capable transcription of the
corresponding exllamav3 inference module's torch fallback, checked on CPU
against that transcription and against the transformers reference math.

| Piece | Where | Oracle |
|---|---|---|
| mHC hyper-connections (Sinkhorn combine, inject, HyperHead weighted / mean) | `training/hyperconnections.py` `mhc_mix/mhc_apply/mhc_head`; `backbone.mhc_spec/hyper_head_spec`; `native_llama._site_in/_site_out`, trunk | `modules/hyperconnections.py` `HyperConnection._mix_unfused/apply_`, `HyperHead.forward` |
| New decoder layout `Embedding ExpandStreams blocks HyperHead RMSNorm Linear` | `backbone._decoder_layout`, `hyper_head` | — |
| Clamped SwiGLU (`swiglu_limit` -> `act_limit`): `min(act(g), L) * clamp(u, -L, L)` | `native_llama._gated_act` (dense, routed, shared) | `exllamav3_ext/activation_kernels.cuh` (activation first, then the clamp; HF clamps g before SiLU, < 5e-4 apart above L) |
| KDA linear attention (per-k-channel safe-gate decay, low-rank f/g gates, sigmoid gated norm) | `training/gdn.py` `kda_beta_g`, `kda_delta_rule_reference`; `native_llama._gdn_forward` kda branch; CUDA bf16 -> `fla.ops.kda.chunk_kda` (installed fla 0.5.1; the vendored copy is forward-only) | `modules/gated_delta_net.py` kda branch, `torch_recurrent_kda` |
| NoPE MLA as dense attention: q_a -> norm -> q_b, latent kv_a -> norm, K/V = c @ w_uk / w_uv | `native_llama._mla_qkv` + the shared attention core (v_head_dim-aware) | `modules/mla_attn.py` (absorbed form; same scores) |
| DSA indexer layers: dense is exact while visible tokens <= `index_topk` (2048) | `block_metadata` -> `qsa_threshold` (existing trunk guard) | `MLAttention._attend` (`sparse = ... > index_topk`) |
| DeepSeek-V4 attention block: sliding-128 + CSA/HCA compressed entries + sink column, V == K, output de-rotation, grouped `wo_a`/`wo_b` | `native_llama._dsv4_forward`, `_dsv4_compress`, `_rope_gptj_trailing` | `modules/dsv4.py` `_forward_nc`, `DSV4Compressor.forward`; transformers `modeling_deepseek_v4` |
| CSA dense-exact bound: T = S // 4 <= index_topk -> S <= 2051 | `qsa_threshold` | `DSV4Attention._forward_nc` (`if T > self.index_topk`) |
| Routers `sqrtsp` / `sqrtsp_hash` (affinity sqrt(softplus), bias for selection only, hash layers by `tid2eid[input_ids]`) | `native_llama._moe_out`; `input_ids` reach the block through the forward ctx | `modules/block_sparse_mlp_routing.py` |
| Validate script: `--stream-experts`, two-process `--save-ref` / `--load-ref` | `training/qlora_validate_native.py` | — |

Target names: GLM-5.3-Flash KDA layers answer to the Llama-style q/k/v/o
(qkv_proj, o_proj) plus explicit `b_proj`, `f_a_proj`, `f_b_proj`, `g_a_proj`,
`g_b_proj`; MLA layers map `q_proj` -> q_a + q_b, `k_proj`/`v_proj` ->
`kv_a_proj_with_mqa`, `o_proj`. DeepSeek-V4: `q_proj` -> wq_a + wq_b,
`k_proj`/`v_proj` -> wkv, `o_proj` -> every wo_a slice + wo_b. Never
trainable: the MLA kv_b tables (not a Linear on the inference side), the DSA /
CSA indexers, the DSV4 compressor projections, the MTP / DSpark heads.

Not supported: sample packing on either model; MLA with a RoPE slice (GLM-4.7-
Flash, DeepSeek-V3, the 500B GLM-5.x) stays rejected; the DSV4 attention has
no flash kernel (eager fp32 scores; at seq 2048 that is ~1.3 GB per layer
under checkpointing).

## Tests

CPU, no GPU/ext: `tests/test_mhc_train.py`, `tests/test_glm53_train.py`,
`tests/test_dsv4_train.py` (all pass; every pre-existing suite still passes;
`tests/test_qlora_grad.py` fails on the untouched tree too with an import
error and is unrelated).

## Box validation

Neither quant fits 2x3090 resident, so the training forward loads with
`--stream-experts` (one card, experts in pinned host RAM), and the inference
oracle runs in a separate process under exllamav3's own CPU expert offload:

```
EXL3_MOE_CPU_OFFLOAD=26 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python training/qlora_validate_native.py --model models/GLM-5.3-Flash-exl3-2.05bpw \
  --parallel split --use-per-device 21 22 --save-ref out/glm53_ref_logits.pt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python training/qlora_validate_native.py --model models/GLM-5.3-Flash-exl3-2.05bpw \
  --stream-experts --load-ref out/glm53_ref_logits.pt --check-backward --skip-dequant-parity
```

GLM-5.3-Flash, turboderp 2.05bpw (80 GB), fp32 compute, eager attention:

| prompt | native | diff | argmax agree | last-token cos | max abs diff |
|---|---|---|---|---|---|
| The capital of France is | Paris | Paris | 83.3% | 0.998178 | 0.93 |
| Once upon a time, there was a | little | little | 100% | 0.998847 | 0.78 |
| Water is made of hydrogen and | oxygen | oxygen | 85.7% | 0.999492 | 0.45 |

Head-slice PASS, backward smoke PASS (loss 5.3752), RESULT PASS. Streaming
parked 42 MoE layers, 70.9 GiB pinned (1728 MiB/layer). The cosine is in the
same band as the earlier MoE validations (Trinity-Nano, Flash-Next: routing
ties flip experts between the two forwards), and here the oracle itself ran
its experts on the CPU worker.

bf16 compute (same reference; fla `chunk_kda` kernel on the 34 KDA layers,
eager attention on the 11 MLA layers since `flash_attn` is not importable in
this venv):

| prompt | diff next-token | argmax agree | last-token cos | max abs diff |
|---|---|---|---|---|
| The capital of France is | Paris (OK) | 100% | 0.999120 | 0.61 |
| Once upon a time, there was a | little (OK) | 100% | 0.997928 | 0.95 |
| Water is made of hydrogen and | oxygen (OK) | 85.7% | 0.998296 | 1.11 |

Backward smoke PASS (loss 5.4718), RESULT PASS. The quant was deleted after
these runs to make room for the DeepSeek download (disk: 19 GB free on /home
with it present).

DeepSeek-V4-Flash: see below.
